# torch2ios

Small lib to serialise Torch7 Networks for iOS. Supported Layers include Fully Connected, Pooling and Convolution Layers at present. The library stores the weights & biases (if any) for each layer necesarry for inference on iOS devices.

Usage is simple: just require 'torch2ios' in your lua script and call saveForiOS(_yourmodel, _destfilename) and a new .t7ios file will be created for use with the KSJNeuralNetwork (Coming Soon) iOS library on the iOS client.

Included is a simple demo of the usage, see xor.lua.

Also added are API to access the binary file format on iOS. See iOSClient. Namespace for the iOS to Torch7 support will follow THES (Torch Embedded Systems) tag.
