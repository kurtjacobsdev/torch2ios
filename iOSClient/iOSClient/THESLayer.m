//
//
//  Created by Kurt Jacobs
//  Copyright © 2016 RandomDudes. All rights reserved.
//
//

#import "THESLayer.h"

NSString * const THES_layer_activations[] = {
@"",@"nn.Linear",@"nn.Convolution",@"nn.MaxPooling",@"nn.AveragePooling",@"nn.Tanh",@"nn.HardTanh",@"nn.LogSigmoid",@"nn.LogSoftMax",@"nn.Sigmoid",@"nn.ReLU"
};

NSString * const THES_tensor_types[] = {
  @"",@"torch.FloatTensor",@"torch.DoubleTensor",@"torch.IntTensor"
};

@implementation THESLayer

- (instancetype)initWithLayerType:(NSNumber *)layerType weightBuffer:(NSValue *)wBuffer weightBufferSize:(NSUInteger)wBufferSize biasBuffer:(NSValue *)bBuffer weightBufferSize:(NSUInteger)bBufferSize
{
  self = [super init];
  if (self)
  {
    self.weightsBuffer = wBuffer;
    self.biasBuffer = bBuffer;
    self.weightsBufferSize = wBufferSize;
    self.biasBufferSize = bBufferSize;
    self.layerType = layerType;
  }
  return self;
}

- (void)freeBuffers
{
  free([self.weightsBuffer pointerValue]);
  free([self.biasBuffer pointerValue]);
}

+ (void)freeLayerBuffers:(NSArray *)layers
{
  [layers enumerateObjectsUsingBlock:^(THESLayer *layer, NSUInteger idx, BOOL * _Nonnull stop)
  {
    [layer freeBuffers];
  }];
}

@end
