	gDio?]O@gDio?]O@!gDio?]O@	?aU?????aU????!?aU????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$gDio?]O@??H.?!??Az?):?;O@Y?	h"lx??*	     0n@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2:??H???!?????Q@)M?O???1???9??P@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism]?C?????!Z2??^;@)+??Χ?1???J1A3@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchn????!s??\; @)n????1s??\; @:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?St$????!jhɬ~@)?St$????1jhɬ~@:Preprocessing2F
Iterator::Model??d?`T??!G(???=@)?I+?v?1lqN?.8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?aU????IOU??s?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??H.?!????H.?!??!??H.?!??      ??!       "      ??!       *      ??!       2	z?):?;O@z?):?;O@!z?):?;O@:      ??!       B      ??!       J	?	h"lx???	h"lx??!?	h"lx??R      ??!       Z	?	h"lx???	h"lx??!?	h"lx??b      ??!       JCPU_ONLYY?aU????b qOU??s?X@