//
//
//  Created by Kurt Jacobs
//  Copyright © 2016 RandomDudes. All rights reserved.
//
//

#import "ViewController.h"
#import "THESDiskFile.h"
#import "THESLayer.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad
{
  [super viewDidLoad];
  
//  NSArray *layers = [THESDiskFile readLayersBinary:[[NSBundle mainBundle] pathForResource:@"ios_xor" ofType:@"t7ios"]];
  NSArray *layers = [THESDiskFile readLayersBinary:[[NSBundle mainBundle] pathForResource:@"mnist_ios" ofType:@"t7ios"]];
  [layers enumerateObjectsUsingBlock:^(THESLayer *layer, NSUInteger idx, BOOL * _Nonnull stop)
  {
    NSLog(@"%@",THES_layer_activations[[layer.layerType integerValue]]);
    NSLog(@"%lu",layer.weightsBufferSize);
    NSLog(@"%lu",layer.biasBufferSize);
  }];
  
  [THESLayer freeLayerBuffers:layers];
}

- (void)didReceiveMemoryWarning
{
  [super didReceiveMemoryWarning];
}

@end
