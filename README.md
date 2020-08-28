    This is the source code and supplementary meterials for paper 'Crowd-wisdom Transferable Bug Finding'
	
Category Structure
------
* train: training workflow of the model
* evaluate: evaluation workflow of the model
* mtl/model_base: the basic model structure
* mtl/model_mtl: MDnet model design
* mtl/sampler_multi_task_trainer: the basic model optimization
* mtl/multi_task_trainer: the model optimization
* mtl/data_prepare and mtl/dataset_reader: data preparation and io
* mtl/config: the parameters setting
* mtl/attention: retrieve the weight of terms in attention layer
* data: the details of experimental results and user studies