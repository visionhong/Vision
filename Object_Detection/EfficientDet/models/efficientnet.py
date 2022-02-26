import torch
from torch import nn
from torch.nn import functional as F

from EfficientDet.models.utils import(
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)


class MBConvBlock(nn.Module):
    '''
    Mobile Inverted Residual Bottleneck Block
    has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    '''

    def __init__(self, block_args, global_params):
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # 0.01
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0<self._block_args.se_ratio <=1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # expand dim
        if self._block_args.expand_ratio != 1:
            # expand_ratio가 6(paper)이라면 1x1 conv로 채널 확장
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, kernel_size=k, stride=s, groups=oup, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase (projection)
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        '''
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        '''

        # Epansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)  # feature_map size를 1 x 1로 만듦
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x  # 각 채널의 중요도(0~1) * 원본

        x = self._bn2(self._project_conv(x))

        # skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        # stride가 1, input output 채널이 같아야 skip connection 가능
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)  # 레이어 drop할지 말지
            x = x + inputs  # skip connection

        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNet, self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Barch norm param
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # RGB
        # number of output channels
        out_channels = round_filters(32, self._global_params)  # stage 1 의 scale에 따른 out_channel
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for i in range(len(self._blocks_args)): # 7번
            # Update block input and output filters based on depth multiplier
            # 현재 network에 맞는 channel수와 반복횟수를 적용
            self._blocks_args[i] = self._blocks_args[i]._replace(
                input_filters = round_filters(
                    self._blocks_args[i].input_filters, self._global_params),
                output_filters = round_filters(
                    self._blocks_args[i].output_filters, self._global_params),
                num_repeat= round_repeats(
                    self._blocks_args[i].num_repeat, self._global_params)
            )

            # 현재 stage의 첫번째 블럭 (stride와 filters에 따라 resolution과 channel이 변함)
            self._blocks.append(MBConvBlock(
                self._blocks_args[i], self._global_params))
            # 현재 stage 블럭의 layers가 2 이상이면 두번째 반복부터는 filter가 고정되야 하고 stride가 1로 되어야 함
            if self._blocks_args[i].num_repeat > 1:
                self._blocks_args[i] = self._blocks_args[i]._replace(
                    input_filters = self._blocks_args[i].output_filters, stride=1)
            for _ in range(self._blocks_args[i].num_repeat-1):  # 두번째 반복부터
                self._blocks.append(MBConvBlock(
                    self._blocks_args[i], self._global_params))

        # Head' efficientdet-d0 : efficientnet-b0
        # 마지막 stage 9
        in_channels = self._blocks_args[len(self._blocks_args)-1].output_filters  # state 8에서의 output filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        ''' Sets swish function as memory effienct (for training) or standard (for export) '''
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        ''' Returns output of the final convolution layer '''
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        P = []
        index =  0
        num_repeat = 0
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # 뒤로갈수록 점점 drop rate가 커짐
            x = block(x, drop_connect_rate=drop_connect_rate)  # MBConvBlock forward
            num_repeat = num_repeat + 1
            if(num_repeat == self._blocks_args[index].num_repeat):
                num_repeat = 0
                index += 1
                P.append(x)  # 각 MBConv stage에서 마지막 layer 값을 리스트에 추가
        return P

    def forward(self, inputs):
        ''' Calls extract_features to extract features, applies final linear later, and returns logits. '''
        P = self.extract_features(inputs)
        return P

    @classmethod  # 매개변수 첫번째 값에 class를 넣어줘야함 staticmethod와 마찬가지로 인스턴스없이 바로 호출 가능
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_claases':num_classes})

        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={
            'num_classes': num_classes})
        load_pretrained_weights(
            model, model_name, load_fc=(num_classes == 1000))

        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
                the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be on of: '+', '.join(valid_models))

    def get_list_features(self):
        list_feature = []
        for idx in range(len(self._blocks_args)):
            list_feature.append(self._blocks_args[idx].output_filters)

        return list_feature


if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b5')
    inputs = torch.randn(4,3,640,640)
    P = model(inputs)
    for idx, p in enumerate(P):
        print('P{}: {}'.format(idx, p.size()))



