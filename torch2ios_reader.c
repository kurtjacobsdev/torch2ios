// Kurt Jacobs
// RandomDudes
// 2016

#include <stdio.h>
#include <stdlib.h>

typedef struct layer_entry 
{ 
	int layertype;
	int datatype_w;
	int datatype_b;
	union
	{
		float *weights_buff_f;
		int *weights_buff_i;
		double *weights_buff_d;
	}weight_buffers;
	union
	{
		float *bias_buff_f;
		int *bias_buff_i;
		double *bias_buff_d;
	}bias_buffers;
	int weight_buffer_size;
	int bias_buffer_size;
} TORCH_LAYER;

const char* layers_and_activations[] = {"","nn.Linear","nn.MaxPooling","nn.AveragePooling","nn.Convolution","nn.Tanh","nn.HardTanh","nn.LogSigmoid","nn.LogSoftMax","nn.Sigmoid","nn.ReLU"};
const char* tensor_types[] = {"","torch.FloatTensor","torch.DoubleTensor","torch.IntTensor"};

void read_bin(char filename [], TORCH_LAYER **layers_pp, int *layers_count)
{
	FILE *file_reader;
	file_reader=fopen(filename,"rb");
	if (!file_reader)
	{
		printf("Unable to open file!");
		return;
	}

	int wc = 0;
	int bc = 0;
	int layertype = 0;
	int datatype_w = 0;
	int datatype_b = 0;
	void *weights = NULL; 
	void *bias = NULL; 

	fread(layers_count,sizeof(int),1,file_reader);

	if (*layers_count <= 0)
	{
		return;
	}

	TORCH_LAYER *layers_array = (TORCH_LAYER *) calloc(*layers_count,sizeof(TORCH_LAYER));

	int lcc = *layers_count;
	for (int i = 0; i < lcc; i++)
	{
		int wc = 0;
		int bc = 0;
		int layertype = 0;
		int datatype_w = 0;
		int datatype_b = 0;
		void *weights = NULL; 
		void *bias = NULL; 

		fread(&layertype,sizeof(int),1,file_reader);
		if (layertype <= 2)
		{
			fread(&datatype_w,sizeof(int),1,file_reader);
			fread(&wc,sizeof(int),1,file_reader);
			if (datatype_w == 1)
			{
				weights = (float *)calloc(wc,sizeof(float));
				fread(weights,sizeof(float),wc,file_reader);
			}
			else if (datatype_w == 2)
			{
				weights = (double *)calloc(wc,sizeof(double));
				fread(weights,sizeof(double),wc,file_reader);
			}
			else if (datatype_w == 3)
			{
				weights = (int *)calloc(wc,sizeof(int));
				fread(weights,sizeof(int),wc,file_reader);
			}
			fread(&datatype_b,sizeof(int),1,file_reader);
			fread(&bc,sizeof(int),1,file_reader);
			if (datatype_w == 1)
			{
				bias = (float *)calloc(bc,sizeof(float));
				fread(bias,sizeof(float),bc,file_reader);
			}
			else if (datatype_w == 2)
			{
				bias = (double *)calloc(bc,sizeof(double));
				fread(bias,sizeof(double),bc,file_reader);
			}
			else if (datatype_w == 3)
			{
				bias = (int *)calloc(bc,sizeof(int));
				fread(bias,sizeof(int),bc,file_reader);
			}
		}

		layers_array[i].layertype = layertype;
		layers_array[i].datatype_w = datatype_w;
		layers_array[i].datatype_b = datatype_b;
		//floatTensor
		if (datatype_w == 1 && datatype_b == 1)
		{
			layers_array[i].weight_buffers.weights_buff_f = (float*)weights;
			layers_array[i].bias_buffers.bias_buff_f = (float*)bias;
		}
			//doubleTensor
		else if (datatype_w == 2 && datatype_b == 2)
		{
			layers_array[i].weight_buffers.weights_buff_d = (double*)weights;
			layers_array[i].bias_buffers.bias_buff_d = (double*)bias;
		}
			//intTensor
		else if (datatype_w == 3 && datatype_b == 3)
		{
			layers_array[i].weight_buffers.weights_buff_i = (int*)weights;
			layers_array[i].bias_buffers.bias_buff_i = (int*)bias;
		}
		layers_array[i].weight_buffer_size = wc;
		layers_array[i].bias_buffer_size = bc;
		layers_count++;	
	}
	*layers_pp = layers_array;

	fclose(file_reader);
}

int main()
{
		TORCH_LAYER ** layers_pp;
		int lc = 0;
		read_bin("ios_xor.t7ios",layers_pp,&lc);
		TORCH_LAYER *layers_p = *layers_pp;

		printf("\n");
		for (int i = 0; i < lc; i ++)
		{
			printf("Layer Type: %s\n", layers_and_activations[layers_p[i].layertype]);	
			if (layers_p[i].datatype_w > 0)
			{
				printf("DataType W: %s\n", tensor_types[layers_p[i].datatype_w]);
			}
			if (layers_p[i].datatype_b > 0)
			{
				printf("DataType B: %s\n", tensor_types[layers_p[i].datatype_b]);
			}
			if (layers_p[i].weight_buffers.weights_buff_f != NULL || layers_p[i].weight_buffers.weights_buff_i != NULL || layers_p[i].weight_buffers.weights_buff_d != NULL)
			{
				printf("Weight Buffer Size:%i\n",layers_p[i].weight_buffer_size);
				if (layers_p[i].datatype_w == 1)
				{
					printf("First Value In Weight Buffer:%f\n",layers_p[i].weight_buffers.weights_buff_f[0]);
				}
				else if (layers_p[i].datatype_w == 2)
				{
					printf("First Value In Weight Buffer:%lf\n",layers_p[i].weight_buffers.weights_buff_d[0]);
				}
				else if (layers_p[i].datatype_w == 3)
				{
					printf("First Value In Weight Buffer:%i\n",layers_p[i].weight_buffers.weights_buff_i[0]);
				}
			}
			if (layers_p[i].bias_buffers.bias_buff_f != NULL || layers_p[i].bias_buffers.bias_buff_i != NULL || layers_p[i].bias_buffers.bias_buff_d != NULL)
			{
				printf("Bias Buffer Size:%i\n",layers_p[i].bias_buffer_size);
				if (layers_p[i].datatype_b == 1)
				{
					printf("First Value In Bias Buffer:%f\n",layers_p[i].bias_buffers.bias_buff_f[0]);
				}
				else if (layers_p[i].datatype_b == 2)
				{
					printf("First Value In Bias Buffer:%lf\n",layers_p[i].bias_buffers.bias_buff_d[0]);
				}
				else if (layers_p[i].datatype_b == 3)
				{
					printf("First Value In Bias Buffer:%i\n",layers_p[i].bias_buffers.bias_buff_i[0]);
				}
			}
			printf("\n");
		}
		free (layers_p);
		
		return 0;
}