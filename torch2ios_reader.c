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
	void *weights;
	int weight_buffer_size;
	void *biases; 
	int bias_buffer_size;
} TORCH_LAYER;

const char* layers_layers_ptivations[] = {"","nn.Linear","nn.MaxPooling","nn.AveragePooling","nn.Convolution","nn.Tanh","nn.HardTanh","nn.LogSigmoid","nn.LogSoftMax","nn.Sigmoid","nn.ReLU"};
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

	int lc_v = *layers_count;
	for (int i = 0; i < lc_v; i++)
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
			weights = (float *)calloc(wc,sizeof(float));
			fread(weights,sizeof(float),wc,file_reader);
			fread(&datatype_b,sizeof(int),1,file_reader);
			fread(&bc,sizeof(int),1,file_reader);
			bias = (float *)calloc(bc,sizeof(float));
			fread(bias,sizeof(int),bc,file_reader);
		}

		layers_array[i].layertype = layertype;
		layers_array[i].datatype_w = datatype_w;
		layers_array[i].datatype_b = datatype_b;
		layers_array[i].weights = weights;
		layers_array[i].weight_buffer_size = wc;
		layers_array[i].biases = bias;
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
			printf("Layer Type: %s\n", layers_layers_ptivations[layers_p[i].layertype]);	
			if (layers_p[i].datatype_w > 0)
			{
				printf("DataType W: %s\n", tensor_types[layers_p[i].datatype_w]);
			}
			if (layers_p[i].datatype_b > 0)
			{
				printf("DataType B: %s\n", tensor_types[layers_p[i].datatype_b]);
			}
			if (layers_p[i].weights != NULL)
			{
				float *w_buffer_p = (float *)layers_p[i].weights;
				printf("Weight Buffer Size:%i\n",layers_p[i].weight_buffer_size);
				printf("First Value In Weight Buffer:%f\n",w_buffer_p[0]);
			}
			if (layers_p[i].biases != NULL)
			{
				float *b_buffer_p = (float *)layers_p[i].biases;
				printf("Bias Buffer Size:%i\n",layers_p[i].bias_buffer_size);
				printf("First Value In Bias Buffer:%f\n",b_buffer_p[0]);
			}
			
			printf("\n");
		}
		free (layers_p);
		
		return 0;
}