//
//
//  Created by Kurt Jacobs
//  Copyright © 2016 RandomDudes. All rights reserved.
//
//

#import "THESDiskFile.h"
#import <UIKit/UIKit.h>

typedef NS_ENUM(NSUInteger, THESDiskFileDataType)
{
  THESDiskFileDataTypeFloat = 1,
  THESDiskFileDataTypeDouble = 2,
  THESDiskFileDataTypeInt = 3
};

@implementation THESDiskFile

+ (NSArray *)readLayersBinary:(NSString *)path
{
  NSMutableArray *layers = [NSMutableArray array];
  
  NSFileHandle *f = [NSFileHandle fileHandleForReadingAtPath:path];
  int layer_count = 0;
  [[f readDataOfLength:sizeof(int)] getBytes:&layer_count length:sizeof(int)];
  int layer_type = 0;
  int datatype_w = 0;
  int weights_len = 0;
  void *weight_buffer;
  int datatype_b = 0;
  int bias_len = 0;
  void *bias_buffer;
  
  for (int i = 0; i < layer_count ; i ++)
  {
    layer_type = 0;
    datatype_w = 0;
    weights_len = 0;
    weight_buffer = NULL;
    datatype_b = 0;
    bias_len = 0;
    bias_buffer = NULL;
    [[f readDataOfLength:sizeof(int)] getBytes:&layer_type length:sizeof(int)];
    if (layer_type <= 2)
    {
      [[f readDataOfLength:sizeof(int)] getBytes:&datatype_w length:sizeof(int)];
      [[f readDataOfLength:sizeof(int)] getBytes:&weights_len length:sizeof(int)];
      if (datatype_w == THESDiskFileDataTypeFloat)
      {
        weight_buffer = (float *)calloc(weights_len,sizeof(float));
      }
      else if (datatype_w == THESDiskFileDataTypeDouble)
      {
        weight_buffer = (double *)calloc(weights_len,sizeof(double));
      }
      else if (datatype_w == THESDiskFileDataTypeInt)
      {
        weight_buffer = (int *)calloc(weights_len,sizeof(int));
      }
      for (int j = 0; j < weights_len; j++)
      {
        if (datatype_w == THESDiskFileDataTypeFloat)
        {
          float readValue;
          [[f readDataOfLength:sizeof(float)] getBytes:&readValue length:sizeof(float)];
          float *castBuffer = weight_buffer;
          castBuffer[j] = readValue;
        }
        else if (datatype_w == THESDiskFileDataTypeDouble)
        {
          double readValue;
          [[f readDataOfLength:sizeof(double)] getBytes:&readValue length:sizeof(double)];
          double *castBuffer = weight_buffer;
          castBuffer[j] = readValue;
        }
        else if(datatype_w == THESDiskFileDataTypeInt)
        {
          int readValue;
          [[f readDataOfLength:sizeof(int)] getBytes:&readValue length:sizeof(int)];
          int *castBuffer = weight_buffer;
          castBuffer[j] = readValue;
        }
      }
      [[f readDataOfLength:sizeof(int)] getBytes:&datatype_b length:sizeof(int)];
      [[f readDataOfLength:sizeof(int)] getBytes:&bias_len length:sizeof(int)];
      
      if (datatype_b == THESDiskFileDataTypeFloat)
      {
        bias_buffer = (float *)calloc(bias_len,sizeof(float));
      }
      else if (datatype_b == THESDiskFileDataTypeDouble)
      {
        bias_buffer = (double *)calloc(bias_len,sizeof(double));
      }
      else if (datatype_b == THESDiskFileDataTypeInt)
      {
        bias_buffer = (int *)calloc(bias_len,sizeof(int));
      }
      for (int j = 0; j < bias_len; j++)
      {
        if (datatype_b == THESDiskFileDataTypeFloat)
        {
          float readValue;
          [[f readDataOfLength:sizeof(float)] getBytes:&readValue length:sizeof(float)];
          float *castBuffer = bias_buffer;
          castBuffer[j] = readValue;
        }
        else if (datatype_b == THESDiskFileDataTypeDouble)
        {
          double readValue;
          [[f readDataOfLength:sizeof(double)] getBytes:&readValue length:sizeof(double)];
          double *castBuffer = bias_buffer;
          castBuffer[j] = readValue;
        }
        else if(datatype_b == THESDiskFileDataTypeInt)
        {
          int readValue;
          [[f readDataOfLength:sizeof(int)] getBytes:&readValue length:sizeof(int)];
          int *castBuffer = bias_buffer;
          castBuffer[j] = readValue;
        }
      }
    }
    
    THESLayer *layer = [[THESLayer alloc] initWithLayerType:@(layer_type) weightBuffer:[NSValue valueWithPointer:weight_buffer] weightBufferSize:weights_len biasBuffer:[NSValue valueWithPointer:bias_buffer] weightBufferSize:bias_len];
    [layers addObject:layer];
  }

  [f closeFile];
  return layers;
}

@end
