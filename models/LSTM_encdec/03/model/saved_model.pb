??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28޼
?
enc_dec_1/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameenc_dec_1/dense_1/kernel
?
,enc_dec_1/dense_1/kernel/Read/ReadVariableOpReadVariableOpenc_dec_1/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
enc_dec_1/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameenc_dec_1/dense_1/bias
}
*enc_dec_1/dense_1/bias/Read/ReadVariableOpReadVariableOpenc_dec_1/dense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
6enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*G
shared_name86enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel
?
Jenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOp6enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel*
_output_shapes
:	?*
dtype0
?
@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Q
shared_nameB@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel
?
Tenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
4enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias
?
Henc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOp4enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias*
_output_shapes	
:?*
dtype0
?
enc_dec_1/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_nameenc_dec_1/lstm_cell_3/kernel
?
0enc_dec_1/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOpenc_dec_1/lstm_cell_3/kernel*
_output_shapes
:	?*
dtype0
?
&enc_dec_1/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*7
shared_name(&enc_dec_1/lstm_cell_3/recurrent_kernel
?
:enc_dec_1/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp&enc_dec_1/lstm_cell_3/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
enc_dec_1/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameenc_dec_1/lstm_cell_3/bias
?
.enc_dec_1/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOpenc_dec_1/lstm_cell_3/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/enc_dec_1/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/enc_dec_1/dense_1/kernel/m
?
3Adam/enc_dec_1/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_dec_1/dense_1/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/enc_dec_1/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/enc_dec_1/dense_1/bias/m
?
1Adam/enc_dec_1/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_dec_1/dense_1/bias/m*
_output_shapes
:*
dtype0
?
=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*N
shared_name?=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/m
?
QAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/m*
_output_shapes
:	?*
dtype0
?
GAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*X
shared_nameIGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/m
?
[Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/m
?
OAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/m/Read/ReadVariableOpReadVariableOp;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/enc_dec_1/lstm_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#Adam/enc_dec_1/lstm_cell_3/kernel/m
?
7Adam/enc_dec_1/lstm_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/enc_dec_1/lstm_cell_3/kernel/m*
_output_shapes
:	?*
dtype0
?
-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*>
shared_name/-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/m
?
AAdam/enc_dec_1/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
!Adam/enc_dec_1/lstm_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/enc_dec_1/lstm_cell_3/bias/m
?
5Adam/enc_dec_1/lstm_cell_3/bias/m/Read/ReadVariableOpReadVariableOp!Adam/enc_dec_1/lstm_cell_3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/enc_dec_1/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!Adam/enc_dec_1/dense_1/kernel/v
?
3Adam/enc_dec_1/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_dec_1/dense_1/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/enc_dec_1/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/enc_dec_1/dense_1/bias/v
?
1Adam/enc_dec_1/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_dec_1/dense_1/bias/v*
_output_shapes
:*
dtype0
?
=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*N
shared_name?=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/v
?
QAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/v*
_output_shapes
:	?*
dtype0
?
GAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*X
shared_nameIGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/v
?
[Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/v
?
OAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/v/Read/ReadVariableOpReadVariableOp;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/enc_dec_1/lstm_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#Adam/enc_dec_1/lstm_cell_3/kernel/v
?
7Adam/enc_dec_1/lstm_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/enc_dec_1/lstm_cell_3/kernel/v*
_output_shapes
:	?*
dtype0
?
-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*>
shared_name/-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/v
?
AAdam/enc_dec_1/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
!Adam/enc_dec_1/lstm_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/enc_dec_1/lstm_cell_3/bias/v
?
5Adam/enc_dec_1/lstm_cell_3/bias/v/Read/ReadVariableOpReadVariableOp!Adam/enc_dec_1/lstm_cell_3/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?5
value?5B?5 B?5
?
	enc_units
	dec_units
encoder_cells
encoder_stacked
encoder_rnn
decoder_cells
	dense
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
 

0
]
	cells
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api

0
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemfmg$mh%mi&mj'mk(ml)mmvnvo$vp%vq&vr'vs(vt)vu
8
$0
%1
&2
'3
(4
)5
6
7
8
$0
%1
&2
'3
(4
)5
6
7
 
?
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
		variables

trainable_variables
regularization_losses
 
?
/
state_size

$kernel
%recurrent_kernel
&bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api

$0
%1
&2

$0
%1
&2
 
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses

90

$0
%1
&2

$0
%1
&2
 
?

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?
@
state_size

'kernel
(recurrent_kernel
)bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
US
VARIABLE_VALUEenc_dec_1/dense_1/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEenc_dec_1/dense_1/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE4enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEenc_dec_1/lstm_cell_3/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&enc_dec_1/lstm_cell_3/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEenc_dec_1/lstm_cell_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

J0
K1
L2
 
 
 

$0
%1
&2

$0
%1
&2
 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
0	variables
1trainable_variables
2regularization_losses
 

0
 
 
 
 

R0
 

0
 
 
 
 

'0
(1
)2

'0
(1
)2
 
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
 
 
 
 
 
4
	Xtotal
	Ycount
Z	variables
[	keras_api
D
	\total
	]count
^
_fn_kwargs
_	variables
`	keras_api
D
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

Z	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

\0
]1

_	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

d	variables
xv
VARIABLE_VALUEAdam/enc_dec_1/dense_1/kernel/mCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/enc_dec_1/dense_1/bias/mAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/enc_dec_1/lstm_cell_3/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/enc_dec_1/lstm_cell_3/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/enc_dec_1/dense_1/kernel/vCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/enc_dec_1/dense_1/bias/vAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/enc_dec_1/lstm_cell_3/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/enc_dec_1/lstm_cell_3/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????	*
dtype0*!
shape:??????????	
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_26enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel4enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/biasenc_dec_1/lstm_cell_3/kernel&enc_dec_1/lstm_cell_3/recurrent_kernelenc_dec_1/lstm_cell_3/biasenc_dec_1/dense_1/kernelenc_dec_1/dense_1/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1494789
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,enc_dec_1/dense_1/kernel/Read/ReadVariableOp*enc_dec_1/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpJenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/Read/ReadVariableOpTenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpHenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/Read/ReadVariableOp0enc_dec_1/lstm_cell_3/kernel/Read/ReadVariableOp:enc_dec_1/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp.enc_dec_1/lstm_cell_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp3Adam/enc_dec_1/dense_1/kernel/m/Read/ReadVariableOp1Adam/enc_dec_1/dense_1/bias/m/Read/ReadVariableOpQAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/m/Read/ReadVariableOp[Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOpOAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/m/Read/ReadVariableOp7Adam/enc_dec_1/lstm_cell_3/kernel/m/Read/ReadVariableOpAAdam/enc_dec_1/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOp5Adam/enc_dec_1/lstm_cell_3/bias/m/Read/ReadVariableOp3Adam/enc_dec_1/dense_1/kernel/v/Read/ReadVariableOp1Adam/enc_dec_1/dense_1/bias/v/Read/ReadVariableOpQAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/v/Read/ReadVariableOp[Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOpOAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/v/Read/ReadVariableOp7Adam/enc_dec_1/lstm_cell_3/kernel/v/Read/ReadVariableOpAAdam/enc_dec_1/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOp5Adam/enc_dec_1/lstm_cell_3/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1496275
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_dec_1/dense_1/kernelenc_dec_1/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate6enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel4enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/biasenc_dec_1/lstm_cell_3/kernel&enc_dec_1/lstm_cell_3/recurrent_kernelenc_dec_1/lstm_cell_3/biastotalcounttotal_1count_1total_2count_2Adam/enc_dec_1/dense_1/kernel/mAdam/enc_dec_1/dense_1/bias/m=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/mGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/m;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/m#Adam/enc_dec_1/lstm_cell_3/kernel/m-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/m!Adam/enc_dec_1/lstm_cell_3/bias/mAdam/enc_dec_1/dense_1/kernel/vAdam/enc_dec_1/dense_1/bias/v=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/vGAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/v;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/v#Adam/enc_dec_1/lstm_cell_3/kernel/v-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/v!Adam/enc_dec_1/lstm_cell_3/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1496390ō
?
?
5__inference_stacked_rnn_cells_1_layer_call_fn_1495231

inputs

states_0_0

states_0_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs
states_0_0
states_0_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493965p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/1
?
?
)__inference_dense_1_layer_call_fn_1495940

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1494332o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?B
?	
while_body_1495846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Y
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0:	?\
Hwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0:
??V
Gwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorW
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?Z
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??T
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	???<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
,while/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Cwhile/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
.while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulwhile_placeholder_2Ewhile/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/addAddV26while/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:08while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
-while/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd-while/stacked_rnn_cells_1/lstm_cell_2/add:z:0Dwhile/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
5while/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+while/stacked_rnn_cells_1/lstm_cell_2/splitSplit>while/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:06while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
-while/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/mulMul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:???????????
*while/stacked_rnn_cells_1/lstm_cell_2/TanhTanh4while/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul1while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0.while/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/add_1AddV2-while/stacked_rnn_cells_1/lstm_cell_2/mul:z:0/while/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
,while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:00while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp=^while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<^while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp>^while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"?
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resourceGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0"?
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2|
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2z
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2~
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
"enc_dec_1_rnn_1_while_cond_1493589<
8enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_loop_counterB
>enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_maximum_iterations%
!enc_dec_1_rnn_1_while_placeholder'
#enc_dec_1_rnn_1_while_placeholder_1'
#enc_dec_1_rnn_1_while_placeholder_2'
#enc_dec_1_rnn_1_while_placeholder_3>
:enc_dec_1_rnn_1_while_less_enc_dec_1_rnn_1_strided_slice_1U
Qenc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_cond_1493589___redundant_placeholder0U
Qenc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_cond_1493589___redundant_placeholder1U
Qenc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_cond_1493589___redundant_placeholder2U
Qenc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_cond_1493589___redundant_placeholder3"
enc_dec_1_rnn_1_while_identity
?
enc_dec_1/rnn_1/while/LessLess!enc_dec_1_rnn_1_while_placeholder:enc_dec_1_rnn_1_while_less_enc_dec_1_rnn_1_strided_slice_1*
T0*
_output_shapes
: k
enc_dec_1/rnn_1/while/IdentityIdentityenc_dec_1/rnn_1/while/Less:z:0*
T0
*
_output_shapes
: "I
enc_dec_1_rnn_1_while_identity'enc_dec_1/rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_rnn_1_layer_call_fn_1495310
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1493881p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
'__inference_rnn_1_layer_call_fn_1495340

inputs
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494264p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs
?U
?
"enc_dec_1_rnn_1_while_body_1493590<
8enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_loop_counterB
>enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_maximum_iterations%
!enc_dec_1_rnn_1_while_placeholder'
#enc_dec_1_rnn_1_while_placeholder_1'
#enc_dec_1_rnn_1_while_placeholder_2'
#enc_dec_1_rnn_1_while_placeholder_3;
7enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_strided_slice_1_0w
senc_dec_1_rnn_1_while_tensorarrayv2read_tensorlistgetitem_enc_dec_1_rnn_1_tensorarrayunstack_tensorlistfromtensor_0i
Venc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0:	?l
Xenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0:
??f
Wenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0:	?"
enc_dec_1_rnn_1_while_identity$
 enc_dec_1_rnn_1_while_identity_1$
 enc_dec_1_rnn_1_while_identity_2$
 enc_dec_1_rnn_1_while_identity_3$
 enc_dec_1_rnn_1_while_identity_4$
 enc_dec_1_rnn_1_while_identity_59
5enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_strided_slice_1u
qenc_dec_1_rnn_1_while_tensorarrayv2read_tensorlistgetitem_enc_dec_1_rnn_1_tensorarrayunstack_tensorlistfromtensorg
Tenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?j
Venc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??d
Uenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	???Lenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?Kenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?Menc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?
Genc_dec_1/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
9enc_dec_1/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsenc_dec_1_rnn_1_while_tensorarrayv2read_tensorlistgetitem_enc_dec_1_rnn_1_tensorarrayunstack_tensorlistfromtensor_0!enc_dec_1_rnn_1_while_placeholderPenc_dec_1/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Kenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpVenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
<enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul@enc_dec_1/rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Senc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Menc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpXenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
>enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMul#enc_dec_1_rnn_1_while_placeholder_2Uenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
9enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/addAddV2Fenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:0Henc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
Lenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpWenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
=enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd=enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add:z:0Tenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Eenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
;enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/splitSplitNenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:0Fenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
=enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoidDenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
?enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1SigmoidDenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
9enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mulMulCenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0#enc_dec_1_rnn_1_while_placeholder_3*
T0*(
_output_shapes
:???????????
:enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/TanhTanhDenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
;enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_1MulAenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0>enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
;enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1AddV2=enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul:z:0?enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
?enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2SigmoidDenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
<enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh?enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
;enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2MulCenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:0@enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
:enc_dec_1/rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#enc_dec_1_rnn_1_while_placeholder_1!enc_dec_1_rnn_1_while_placeholder?enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:???]
enc_dec_1/rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
enc_dec_1/rnn_1/while/addAddV2!enc_dec_1_rnn_1_while_placeholder$enc_dec_1/rnn_1/while/add/y:output:0*
T0*
_output_shapes
: _
enc_dec_1/rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
enc_dec_1/rnn_1/while/add_1AddV28enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_loop_counter&enc_dec_1/rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
enc_dec_1/rnn_1/while/IdentityIdentityenc_dec_1/rnn_1/while/add_1:z:0^enc_dec_1/rnn_1/while/NoOp*
T0*
_output_shapes
: ?
 enc_dec_1/rnn_1/while/Identity_1Identity>enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_while_maximum_iterations^enc_dec_1/rnn_1/while/NoOp*
T0*
_output_shapes
: ?
 enc_dec_1/rnn_1/while/Identity_2Identityenc_dec_1/rnn_1/while/add:z:0^enc_dec_1/rnn_1/while/NoOp*
T0*
_output_shapes
: ?
 enc_dec_1/rnn_1/while/Identity_3IdentityJenc_dec_1/rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^enc_dec_1/rnn_1/while/NoOp*
T0*
_output_shapes
: ?
 enc_dec_1/rnn_1/while/Identity_4Identity?enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0^enc_dec_1/rnn_1/while/NoOp*
T0*(
_output_shapes
:???????????
 enc_dec_1/rnn_1/while/Identity_5Identity?enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0^enc_dec_1/rnn_1/while/NoOp*
T0*(
_output_shapes
:???????????
enc_dec_1/rnn_1/while/NoOpNoOpM^enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpL^enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpN^enc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "p
5enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_strided_slice_17enc_dec_1_rnn_1_while_enc_dec_1_rnn_1_strided_slice_1_0"I
enc_dec_1_rnn_1_while_identity'enc_dec_1/rnn_1/while/Identity:output:0"M
 enc_dec_1_rnn_1_while_identity_1)enc_dec_1/rnn_1/while/Identity_1:output:0"M
 enc_dec_1_rnn_1_while_identity_2)enc_dec_1/rnn_1/while/Identity_2:output:0"M
 enc_dec_1_rnn_1_while_identity_3)enc_dec_1/rnn_1/while/Identity_3:output:0"M
 enc_dec_1_rnn_1_while_identity_4)enc_dec_1/rnn_1/while/Identity_4:output:0"M
 enc_dec_1_rnn_1_while_identity_5)enc_dec_1/rnn_1/while/Identity_5:output:0"?
Uenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resourceWenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0"?
Venc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceXenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Tenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceVenc_dec_1_rnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0"?
qenc_dec_1_rnn_1_while_tensorarrayv2read_tensorlistgetitem_enc_dec_1_rnn_1_tensorarrayunstack_tensorlistfromtensorsenc_dec_1_rnn_1_while_tensorarrayv2read_tensorlistgetitem_enc_dec_1_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
Lenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpLenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2?
Kenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpKenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2?
Menc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpMenc_dec_1/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
5__inference_stacked_rnn_cells_1_layer_call_fn_1495214

inputs

states_0_0

states_0_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs
states_0_0
states_0_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493797p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/1
?:
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1493881

inputs.
stacked_rnn_cells_1_1493798:	?/
stacked_rnn_cells_1_1493800:
??*
stacked_rnn_cells_1_1493802:	?
identity

identity_1

identity_2??+stacked_rnn_cells_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_1_1493798stacked_rnn_cells_1_1493800stacked_rnn_cells_1_1493802*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493797n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_1_1493798stacked_rnn_cells_1_1493800stacked_rnn_cells_1_1493802*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1493811*
condR
while_cond_1493810*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????|
NoOpNoOp,^stacked_rnn_cells_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+stacked_rnn_cells_1/StatefulPartitionedCall+stacked_rnn_cells_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1496048

inputs
states_0
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?

?
+__inference_enc_dec_1_layer_call_fn_1494811
inputs_0
inputs_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494342s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
while_cond_1495557
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1495557___redundant_placeholder05
1while_while_cond_1495557___redundant_placeholder15
1while_while_cond_1495557___redundant_placeholder25
1while_while_cond_1495557___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493965

inputs

states
states_1&
lstm_cell_2_1493953:	?'
lstm_cell_2_1493955:
??"
lstm_cell_2_1493957:	?
identity

identity_1

identity_2??#lstm_cell_2/StatefulPartitionedCall?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallinputsstatesstates_1lstm_cell_2_1493953lstm_cell_2_1493955lstm_cell_2_1493957*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1493952|
IdentityIdentity,lstm_cell_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????~

Identity_1Identity,lstm_cell_2/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????~

Identity_2Identity,lstm_cell_2/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????l
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?U
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495643
inputs_0Q
>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?T
@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??N
?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
&stacked_rnn_cells_1/lstm_cell_2/MatMulMatMulstrided_slice_2:output:0=stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulzeros:output:0?stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/addAddV20stacked_rnn_cells_1/lstm_cell_2/MatMul:product:02stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd'stacked_rnn_cells_1/lstm_cell_2/add:z:0>stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
%stacked_rnn_cells_1/lstm_cell_2/splitSplit8stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:00stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
'stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/mulMul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
$stacked_rnn_cells_1/lstm_cell_2/TanhTanh.stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_1Mul+stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0(stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/add_1AddV2'stacked_rnn_cells_1/lstm_cell_2/mul:z:0)stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
&stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh)stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_2Mul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:0*stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1495558*
condR
while_cond_1495557*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp7^stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6^stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp8^stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2p
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2n
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2r
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
-__inference_lstm_cell_3_layer_call_fn_1496082

inputs
states_0
states_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494425p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?	
?
rnn_1_while_cond_1494892(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3*
&rnn_1_while_less_rnn_1_strided_slice_1A
=rnn_1_while_rnn_1_while_cond_1494892___redundant_placeholder0A
=rnn_1_while_rnn_1_while_cond_1494892___redundant_placeholder1A
=rnn_1_while_rnn_1_while_cond_1494892___redundant_placeholder2A
=rnn_1_while_rnn_1_while_cond_1494892___redundant_placeholder3
rnn_1_while_identity
z
rnn_1/while/LessLessrnn_1_while_placeholder&rnn_1_while_less_rnn_1_strided_slice_1*
T0*
_output_shapes
: W
rnn_1/while/IdentityIdentityrnn_1/while/Less:z:0*
T0
*
_output_shapes
: "5
rnn_1_while_identityrnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_lstm_cell_2_layer_call_fn_1495984

inputs
states_0
states_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1493952p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494342

inputs
inputs_1 
rnn_1_1494265:	?!
rnn_1_1494267:
??
rnn_1_1494269:	?&
lstm_cell_3_1494313:	?'
lstm_cell_3_1494315:
??"
lstm_cell_3_1494317:	?"
dense_1_1494333:	?
dense_1_1494335:
identity??dense_1/StatefulPartitionedCall?#lstm_cell_3/StatefulPartitionedCall?rnn_1/StatefulPartitionedCall?
rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputsrnn_1_1494265rnn_1_1494267rnn_1_1494269*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494264h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0&rnn_1/StatefulPartitionedCall:output:1&rnn_1/StatefulPartitionedCall:output:2lstm_cell_3_1494313lstm_cell_3_1494315lstm_cell_3_1494317*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494312?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,lstm_cell_3/StatefulPartitionedCall:output:0dense_1_1494333dense_1_1494335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1494332v
stackPack(dense_1/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^lstm_cell_3/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
+__inference_enc_dec_1_layer_call_fn_1494685
input_1
input_2
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494644s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????	
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?$
?
while_body_1494059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
#while_stacked_rnn_cells_1_1494083_0:	?7
#while_stacked_rnn_cells_1_1494085_0:
??2
#while_stacked_rnn_cells_1_1494087_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
!while_stacked_rnn_cells_1_1494083:	?5
!while_stacked_rnn_cells_1_1494085:
??0
!while_stacked_rnn_cells_1_1494087:	???1while/stacked_rnn_cells_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3#while_stacked_rnn_cells_1_1494083_0#while_stacked_rnn_cells_1_1494085_0#while_stacked_rnn_cells_1_1494087_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493965?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp2^while/stacked_rnn_cells_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"H
!while_stacked_rnn_cells_1_1494083#while_stacked_rnn_cells_1_1494083_0"H
!while_stacked_rnn_cells_1_1494085#while_stacked_rnn_cells_1_1494085_0"H
!while_stacked_rnn_cells_1_1494087#while_stacked_rnn_cells_1_1494087_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1while/stacked_rnn_cells_1/StatefulPartitionedCall1while/stacked_rnn_cells_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1495263

inputs

states_0_0

states_0_1=
*lstm_cell_2_matmul_readvariableop_resource:	?@
,lstm_cell_2_matmul_1_readvariableop_resource:
??:
+lstm_cell_2_biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/MatMulMatMulinputs)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_2/MatMul_1MatMul
states_0_0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????p
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0
states_0_1*
T0*(
_output_shapes
:??????????g
lstm_cell_2/TanhTanhlstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????z
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????e
IdentityIdentitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_1Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_2Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/1
?I
?
rnn_1_while_body_1494893(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3'
#rnn_1_while_rnn_1_strided_slice_1_0c
_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0_
Lrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0:	?b
Nrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0:
??\
Mrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0:	?
rnn_1_while_identity
rnn_1_while_identity_1
rnn_1_while_identity_2
rnn_1_while_identity_3
rnn_1_while_identity_4
rnn_1_while_identity_5%
!rnn_1_while_rnn_1_strided_slice_1a
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor]
Jrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?`
Lrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??Z
Krnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	???Brnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?Arnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?Crnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0rnn_1_while_placeholderFrnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Arnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpLrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
2rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Irnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Crnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpNrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
4rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulrnn_1_while_placeholder_2Krnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/addAddV2<rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:0>rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
Brnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpMrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
3rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd3rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add:z:0Jrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
;rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/splitSplitDrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:0<rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
3rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mulMul9rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0rnn_1_while_placeholder_3*
T0*(
_output_shapes
:???????????
0rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/TanhTanh:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul7rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:04rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1AddV23rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul:z:05rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
2rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul9rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:06rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
0rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_1_while_placeholder_1rnn_1_while_placeholder5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:???S
rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
rnn_1/while/addAddV2rnn_1_while_placeholderrnn_1/while/add/y:output:0*
T0*
_output_shapes
: U
rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_1/while/add_1AddV2$rnn_1_while_rnn_1_while_loop_counterrnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
rnn_1/while/IdentityIdentityrnn_1/while/add_1:z:0^rnn_1/while/NoOp*
T0*
_output_shapes
: ?
rnn_1/while/Identity_1Identity*rnn_1_while_rnn_1_while_maximum_iterations^rnn_1/while/NoOp*
T0*
_output_shapes
: k
rnn_1/while/Identity_2Identityrnn_1/while/add:z:0^rnn_1/while/NoOp*
T0*
_output_shapes
: ?
rnn_1/while/Identity_3Identity@rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_1/while/NoOp*
T0*
_output_shapes
: ?
rnn_1/while/Identity_4Identity5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0^rnn_1/while/NoOp*
T0*(
_output_shapes
:???????????
rnn_1/while/Identity_5Identity5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0^rnn_1/while/NoOp*
T0*(
_output_shapes
:???????????
rnn_1/while/NoOpNoOpC^rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpB^rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpD^rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "5
rnn_1_while_identityrnn_1/while/Identity:output:0"9
rnn_1_while_identity_1rnn_1/while/Identity_1:output:0"9
rnn_1_while_identity_2rnn_1/while/Identity_2:output:0"9
rnn_1_while_identity_3rnn_1/while/Identity_3:output:0"9
rnn_1_while_identity_4rnn_1/while/Identity_4:output:0"9
rnn_1_while_identity_5rnn_1/while/Identity_5:output:0"H
!rnn_1_while_rnn_1_strided_slice_1#rnn_1_while_rnn_1_strided_slice_1_0"?
Krnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resourceMrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0"?
Lrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceNrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Jrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceLrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0"?
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
Brnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpBrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2?
Arnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpArnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2?
Crnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpCrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_1494499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
#while_stacked_rnn_cells_1_1494523_0:	?7
#while_stacked_rnn_cells_1_1494525_0:
??2
#while_stacked_rnn_cells_1_1494527_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
!while_stacked_rnn_cells_1_1494523:	?5
!while_stacked_rnn_cells_1_1494525:
??0
!while_stacked_rnn_cells_1_1494527:	???1while/stacked_rnn_cells_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3#while_stacked_rnn_cells_1_1494523_0#while_stacked_rnn_cells_1_1494525_0#while_stacked_rnn_cells_1_1494527_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493965?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp2^while/stacked_rnn_cells_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"H
!while_stacked_rnn_cells_1_1494523#while_stacked_rnn_cells_1_1494523_0"H
!while_stacked_rnn_cells_1_1494525#while_stacked_rnn_cells_1_1494525_0"H
!while_stacked_rnn_cells_1_1494527#while_stacked_rnn_cells_1_1494527_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1while/stacked_rnn_cells_1/StatefulPartitionedCall1while/stacked_rnn_cells_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1495197
inputs_0
inputs_1W
Drnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?Z
Frnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??T
Ernn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	?=
*lstm_cell_3_matmul_readvariableop_resource:	?@
,lstm_cell_3_matmul_1_readvariableop_resource:
??:
+lstm_cell_3_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?"lstm_cell_3/BiasAdd/ReadVariableOp?!lstm_cell_3/MatMul/ReadVariableOp?#lstm_cell_3/MatMul_1/ReadVariableOp?<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?rnn_1/whileC
rnn_1/ShapeShapeinputs_0*
T0*
_output_shapes
:c
rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_sliceStridedSlicernn_1/Shape:output:0"rnn_1/strided_slice/stack:output:0$rnn_1/strided_slice/stack_1:output:0$rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
rnn_1/zeros/packedPackrnn_1/strided_slice:output:0rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_1/zerosFillrnn_1/zeros/packed:output:0rnn_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Y
rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
rnn_1/zeros_1/packedPackrnn_1/strided_slice:output:0rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:X
rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
rnn_1/zeros_1Fillrnn_1/zeros_1/packed:output:0rnn_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????i
rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rnn_1/transpose	Transposeinputs_0rnn_1/transpose/perm:output:0*
T0*,
_output_shapes
:?	?????????P
rnn_1/Shape_1Shapernn_1/transpose:y:0*
T0*
_output_shapes
:e
rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_slice_1StridedSlicernn_1/Shape_1:output:0$rnn_1/strided_slice_1/stack:output:0&rnn_1/strided_slice_1/stack_1:output:0&rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn_1/TensorArrayV2TensorListReserve*rnn_1/TensorArrayV2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
-rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_1/transpose:y:0Drnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???e
rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_slice_2StridedSlicernn_1/transpose:y:0$rnn_1/strided_slice_2/stack:output:0&rnn_1/strided_slice_2/stack_1:output:0&rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpDrnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
,rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMulrnn_1/strided_slice_2:output:0Crnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpFrnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
.rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulrnn_1/zeros:output:0Ernn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)rnn_1/stacked_rnn_cells_1/lstm_cell_2/addAddV26rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:08rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpErnn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
-rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd-rnn_1/stacked_rnn_cells_1/lstm_cell_2/add:z:0Drnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
5rnn_1/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/splitSplit>rnn_1/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:06rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
-rnn_1/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
)rnn_1/stacked_rnn_cells_1/lstm_cell_2/mulMul3rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0rnn_1/zeros_1:output:0*
T0*(
_output_shapes
:???????????
*rnn_1/stacked_rnn_cells_1/lstm_cell_2/TanhTanh4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul1rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0.rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/add_1AddV2-rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul:z:0/rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
,rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh/rnn_1/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul3rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:00rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????t
#rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
rnn_1/TensorArrayV2_1TensorListReserve,rnn_1/TensorArrayV2_1/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???L

rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Z
rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
rnn_1/whileWhile!rnn_1/while/loop_counter:output:0'rnn_1/while/maximum_iterations:output:0rnn_1/time:output:0rnn_1/TensorArrayV2_1:handle:0rnn_1/zeros:output:0rnn_1/zeros_1:output:0rnn_1/strided_slice_1:output:0=rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Drnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceFrnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceErnn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
rnn_1_while_body_1495075*$
condR
rnn_1_while_cond_1495074*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
(rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_1/while:output:3?rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:?	??????????*
element_dtype0n
rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????g
rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_slice_3StridedSlice1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_1/strided_slice_3/stack:output:0&rnn_1/strided_slice_3/stack_1:output:0&rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskk
rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn_1/transpose_1	Transpose1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0rnn_1/transpose_1/perm:output:0*
T0*-
_output_shapes
:??????????	?h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputs_0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_3/MatMulMatMulconcat:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/MatMul_1MatMulrnn_1/while:output:4+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????z
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0rnn_1/while:output:5*
T0*(
_output_shapes
:??????????g
lstm_cell_3/TanhTanhlstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????z
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????y
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_1/MatMulMatMullstm_cell_3/mul_2:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
stackPackdense_1/BiasAdd:output:0*
N*
T0*+
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp=^rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<^rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp>^rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp^rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2|
<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2z
;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2~
=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp2
rnn_1/whilernn_1/while:V R
,
_output_shapes
:??????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1496146

inputs
states_0
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?$
?
while_body_1493811
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
#while_stacked_rnn_cells_1_1493835_0:	?7
#while_stacked_rnn_cells_1_1493837_0:
??2
#while_stacked_rnn_cells_1_1493839_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
!while_stacked_rnn_cells_1_1493835:	?5
!while_stacked_rnn_cells_1_1493837:
??0
!while_stacked_rnn_cells_1_1493839:	???1while/stacked_rnn_cells_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3#while_stacked_rnn_cells_1_1493835_0#while_stacked_rnn_cells_1_1493837_0#while_stacked_rnn_cells_1_1493839_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493797?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp2^while/stacked_rnn_cells_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"H
!while_stacked_rnn_cells_1_1493835#while_stacked_rnn_cells_1_1493835_0"H
!while_stacked_rnn_cells_1_1493837#while_stacked_rnn_cells_1_1493837_0"H
!while_stacked_rnn_cells_1_1493839#while_stacked_rnn_cells_1_1493839_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1while/stacked_rnn_cells_1/StatefulPartitionedCall1while/stacked_rnn_cells_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
+__inference_enc_dec_1_layer_call_fn_1494833
inputs_0
inputs_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494644s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:??????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494425

inputs

states
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?	
?
rnn_1_while_cond_1495074(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3*
&rnn_1_while_less_rnn_1_strided_slice_1A
=rnn_1_while_rnn_1_while_cond_1495074___redundant_placeholder0A
=rnn_1_while_rnn_1_while_cond_1495074___redundant_placeholder1A
=rnn_1_while_rnn_1_while_cond_1495074___redundant_placeholder2A
=rnn_1_while_rnn_1_while_cond_1495074___redundant_placeholder3
rnn_1_while_identity
z
rnn_1/while/LessLessrnn_1_while_placeholder&rnn_1_while_less_rnn_1_strided_slice_1*
T0*
_output_shapes
: W
rnn_1/while/IdentityIdentityrnn_1/while/Less:z:0*
T0
*
_output_shapes
: "5
rnn_1_while_identityrnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494759
input_1
input_2 
rnn_1_1494726:	?!
rnn_1_1494728:
??
rnn_1_1494730:	?&
lstm_cell_3_1494741:	?'
lstm_cell_3_1494743:
??"
lstm_cell_3_1494745:	?"
dense_1_1494750:	?
dense_1_1494752:
identity??dense_1/StatefulPartitionedCall?#lstm_cell_3/StatefulPartitionedCall?rnn_1/StatefulPartitionedCall?
rnn_1/StatefulPartitionedCallStatefulPartitionedCallinput_1rnn_1_1494726rnn_1_1494728rnn_1_1494730*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494569h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0input_2concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0&rnn_1/StatefulPartitionedCall:output:1&rnn_1/StatefulPartitionedCall:output:2lstm_cell_3_1494741lstm_cell_3_1494743lstm_cell_3_1494745*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494425?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,lstm_cell_3/StatefulPartitionedCall:output:0dense_1_1494750dense_1_1494752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1494332v
stackPack(dense_1/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^lstm_cell_3/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????	
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?$
?
while_body_1494194
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
#while_stacked_rnn_cells_1_1494218_0:	?7
#while_stacked_rnn_cells_1_1494220_0:
??2
#while_stacked_rnn_cells_1_1494222_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
!while_stacked_rnn_cells_1_1494218:	?5
!while_stacked_rnn_cells_1_1494220:
??0
!while_stacked_rnn_cells_1_1494222:	???1while/stacked_rnn_cells_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3#while_stacked_rnn_cells_1_1494218_0#while_stacked_rnn_cells_1_1494220_0#while_stacked_rnn_cells_1_1494222_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493797?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity:while/stacked_rnn_cells_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp2^while/stacked_rnn_cells_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"H
!while_stacked_rnn_cells_1_1494218#while_stacked_rnn_cells_1_1494218_0"H
!while_stacked_rnn_cells_1_1494220#while_stacked_rnn_cells_1_1494220_0"H
!while_stacked_rnn_cells_1_1494222#while_stacked_rnn_cells_1_1494222_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1while/stacked_rnn_cells_1/StatefulPartitionedCall1while/stacked_rnn_cells_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1494058
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1494058___redundant_placeholder05
1while_while_cond_1494058___redundant_placeholder15
1while_while_cond_1494058___redundant_placeholder25
1while_while_cond_1494058___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1495701
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1495701___redundant_placeholder05
1while_while_cond_1495701___redundant_placeholder15
1while_while_cond_1495701___redundant_placeholder25
1while_while_cond_1495701___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
%__inference_signature_wrapper_1494789
input_1
input_2
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_1493712s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????	
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?:
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494569

inputs.
stacked_rnn_cells_1_1494486:	?/
stacked_rnn_cells_1_1494488:
??*
stacked_rnn_cells_1_1494490:	?
identity

identity_1

identity_2??+stacked_rnn_cells_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_1_1494486stacked_rnn_cells_1_1494488stacked_rnn_cells_1_1494490*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493965n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_1_1494486stacked_rnn_cells_1_1494488stacked_rnn_cells_1_1494490*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1494499*
condR
while_cond_1494498*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:?	??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:??????????	?h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????|
NoOpNoOp,^stacked_rnn_cells_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????	: : : 2Z
+stacked_rnn_cells_1/StatefulPartitionedCall+stacked_rnn_cells_1/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1496016

inputs
states_0
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?O
?
 __inference__traced_save_1496275
file_prefix7
3savev2_enc_dec_1_dense_1_kernel_read_readvariableop5
1savev2_enc_dec_1_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopU
Qsavev2_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_read_readvariableop_
[savev2_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_read_readvariableopS
Osavev2_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_read_readvariableop;
7savev2_enc_dec_1_lstm_cell_3_kernel_read_readvariableopE
Asavev2_enc_dec_1_lstm_cell_3_recurrent_kernel_read_readvariableop9
5savev2_enc_dec_1_lstm_cell_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop>
:savev2_adam_enc_dec_1_dense_1_kernel_m_read_readvariableop<
8savev2_adam_enc_dec_1_dense_1_bias_m_read_readvariableop\
Xsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_m_read_readvariableopf
bsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_m_read_readvariableopZ
Vsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_m_read_readvariableopB
>savev2_adam_enc_dec_1_lstm_cell_3_kernel_m_read_readvariableopL
Hsavev2_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_m_read_readvariableop@
<savev2_adam_enc_dec_1_lstm_cell_3_bias_m_read_readvariableop>
:savev2_adam_enc_dec_1_dense_1_kernel_v_read_readvariableop<
8savev2_adam_enc_dec_1_dense_1_bias_v_read_readvariableop\
Xsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_v_read_readvariableopf
bsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_v_read_readvariableopZ
Vsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_v_read_readvariableopB
>savev2_adam_enc_dec_1_lstm_cell_3_kernel_v_read_readvariableopL
Hsavev2_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_v_read_readvariableop@
<savev2_adam_enc_dec_1_lstm_cell_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_enc_dec_1_dense_1_kernel_read_readvariableop1savev2_enc_dec_1_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopQsavev2_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_read_readvariableop[savev2_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_read_readvariableopOsavev2_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_read_readvariableop7savev2_enc_dec_1_lstm_cell_3_kernel_read_readvariableopAsavev2_enc_dec_1_lstm_cell_3_recurrent_kernel_read_readvariableop5savev2_enc_dec_1_lstm_cell_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop:savev2_adam_enc_dec_1_dense_1_kernel_m_read_readvariableop8savev2_adam_enc_dec_1_dense_1_bias_m_read_readvariableopXsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_m_read_readvariableopbsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_m_read_readvariableopVsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_m_read_readvariableop>savev2_adam_enc_dec_1_lstm_cell_3_kernel_m_read_readvariableopHsavev2_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_m_read_readvariableop<savev2_adam_enc_dec_1_lstm_cell_3_bias_m_read_readvariableop:savev2_adam_enc_dec_1_dense_1_kernel_v_read_readvariableop8savev2_adam_enc_dec_1_dense_1_bias_v_read_readvariableopXsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_v_read_readvariableopbsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_v_read_readvariableopVsavev2_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_v_read_readvariableop>savev2_adam_enc_dec_1_lstm_cell_3_kernel_v_read_readvariableopHsavev2_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_v_read_readvariableop<savev2_adam_enc_dec_1_lstm_cell_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:: : : : : :	?:
??:?:	?:
??:?: : : : : : :	?::	?:
??:?:	?:
??:?:	?::	?:
??:?:	?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:%!!

_output_shapes
:	?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:$

_output_shapes
: 
?
?
-__inference_lstm_cell_2_layer_call_fn_1495967

inputs
states_0
states_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1493784p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494644

inputs
inputs_1 
rnn_1_1494611:	?!
rnn_1_1494613:
??
rnn_1_1494615:	?&
lstm_cell_3_1494626:	?'
lstm_cell_3_1494628:
??"
lstm_cell_3_1494630:	?"
dense_1_1494635:	?
dense_1_1494637:
identity??dense_1/StatefulPartitionedCall?#lstm_cell_3/StatefulPartitionedCall?rnn_1/StatefulPartitionedCall?
rnn_1/StatefulPartitionedCallStatefulPartitionedCallinputsrnn_1_1494611rnn_1_1494613rnn_1_1494615*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494569h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0&rnn_1/StatefulPartitionedCall:output:1&rnn_1/StatefulPartitionedCall:output:2lstm_cell_3_1494626lstm_cell_3_1494628lstm_cell_3_1494630*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494425?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,lstm_cell_3/StatefulPartitionedCall:output:0dense_1_1494635dense_1_1494637*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1494332v
stackPack(dense_1/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^lstm_cell_3/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_rnn_1_layer_call_fn_1495355

inputs
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494569p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1495295

inputs

states_0_0

states_0_1=
*lstm_cell_2_matmul_readvariableop_resource:	?@
,lstm_cell_2_matmul_1_readvariableop_resource:
??:
+lstm_cell_2_biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??"lstm_cell_2/BiasAdd/ReadVariableOp?!lstm_cell_2/MatMul/ReadVariableOp?#lstm_cell_2/MatMul_1/ReadVariableOp?
!lstm_cell_2/MatMul/ReadVariableOpReadVariableOp*lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/MatMulMatMulinputs)lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_2/MatMul_1MatMul
states_0_0+lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/MatMul:product:0lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
"lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/add:z:0*lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_2/SigmoidSigmoidlstm_cell_2/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_1Sigmoidlstm_cell_2/split:output:1*
T0*(
_output_shapes
:??????????p
lstm_cell_2/mulMullstm_cell_2/Sigmoid_1:y:0
states_0_1*
T0*(
_output_shapes
:??????????g
lstm_cell_2/TanhTanhlstm_cell_2/split:output:2*
T0*(
_output_shapes
:??????????z
lstm_cell_2/mul_1Mullstm_cell_2/Sigmoid:y:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????y
lstm_cell_2/add_1AddV2lstm_cell_2/mul:z:0lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_2/Sigmoid_2Sigmoidlstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/Sigmoid_2:y:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????e
IdentityIdentitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_1Identitylstm_cell_2/mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????g

Identity_2Identitylstm_cell_2/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp#^lstm_cell_2/BiasAdd/ReadVariableOp"^lstm_cell_2/MatMul/ReadVariableOp$^lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2H
"lstm_cell_2/BiasAdd/ReadVariableOp"lstm_cell_2/BiasAdd/ReadVariableOp2F
!lstm_cell_2/MatMul/ReadVariableOp!lstm_cell_2/MatMul/ReadVariableOp2J
#lstm_cell_2/MatMul_1/ReadVariableOp#lstm_cell_2/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:??????????
$
_user_specified_name
states/0/1
?U
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495499
inputs_0Q
>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?T
@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??N
?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
&stacked_rnn_cells_1/lstm_cell_2/MatMulMatMulstrided_slice_2:output:0=stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulzeros:output:0?stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/addAddV20stacked_rnn_cells_1/lstm_cell_2/MatMul:product:02stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd'stacked_rnn_cells_1/lstm_cell_2/add:z:0>stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
%stacked_rnn_cells_1/lstm_cell_2/splitSplit8stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:00stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
'stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/mulMul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
$stacked_rnn_cells_1/lstm_cell_2/TanhTanh.stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_1Mul+stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0(stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/add_1AddV2'stacked_rnn_cells_1/lstm_cell_2/mul:z:0)stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
&stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh)stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_2Mul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:0*stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1495414*
condR
while_cond_1495413*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp7^stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6^stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp8^stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2p
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2n
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2r
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493797

inputs

states
states_1&
lstm_cell_2_1493785:	?'
lstm_cell_2_1493787:
??"
lstm_cell_2_1493789:	?
identity

identity_1

identity_2??#lstm_cell_2/StatefulPartitionedCall?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallinputsstatesstates_1lstm_cell_2_1493785lstm_cell_2_1493787lstm_cell_2_1493789*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1493784|
IdentityIdentity,lstm_cell_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????~

Identity_1Identity,lstm_cell_2/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????~

Identity_2Identity,lstm_cell_2/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????l
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494312

inputs

states
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
ܕ
?	
"__inference__wrapped_model_1493712
input_1
input_2a
Nenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?d
Penc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??^
Oenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	?G
4enc_dec_1_lstm_cell_3_matmul_readvariableop_resource:	?J
6enc_dec_1_lstm_cell_3_matmul_1_readvariableop_resource:
??D
5enc_dec_1_lstm_cell_3_biasadd_readvariableop_resource:	?C
0enc_dec_1_dense_1_matmul_readvariableop_resource:	??
1enc_dec_1_dense_1_biasadd_readvariableop_resource:
identity??(enc_dec_1/dense_1/BiasAdd/ReadVariableOp?'enc_dec_1/dense_1/MatMul/ReadVariableOp?,enc_dec_1/lstm_cell_3/BiasAdd/ReadVariableOp?+enc_dec_1/lstm_cell_3/MatMul/ReadVariableOp?-enc_dec_1/lstm_cell_3/MatMul_1/ReadVariableOp?Fenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?Eenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?Genc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?enc_dec_1/rnn_1/whileL
enc_dec_1/rnn_1/ShapeShapeinput_1*
T0*
_output_shapes
:m
#enc_dec_1/rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%enc_dec_1/rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%enc_dec_1/rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
enc_dec_1/rnn_1/strided_sliceStridedSliceenc_dec_1/rnn_1/Shape:output:0,enc_dec_1/rnn_1/strided_slice/stack:output:0.enc_dec_1/rnn_1/strided_slice/stack_1:output:0.enc_dec_1/rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
enc_dec_1/rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
enc_dec_1/rnn_1/zeros/packedPack&enc_dec_1/rnn_1/strided_slice:output:0'enc_dec_1/rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
enc_dec_1/rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
enc_dec_1/rnn_1/zerosFill%enc_dec_1/rnn_1/zeros/packed:output:0$enc_dec_1/rnn_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????c
 enc_dec_1/rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
enc_dec_1/rnn_1/zeros_1/packedPack&enc_dec_1/rnn_1/strided_slice:output:0)enc_dec_1/rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
enc_dec_1/rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
enc_dec_1/rnn_1/zeros_1Fill'enc_dec_1/rnn_1/zeros_1/packed:output:0&enc_dec_1/rnn_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????s
enc_dec_1/rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
enc_dec_1/rnn_1/transpose	Transposeinput_1'enc_dec_1/rnn_1/transpose/perm:output:0*
T0*,
_output_shapes
:?	?????????d
enc_dec_1/rnn_1/Shape_1Shapeenc_dec_1/rnn_1/transpose:y:0*
T0*
_output_shapes
:o
%enc_dec_1/rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'enc_dec_1/rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'enc_dec_1/rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
enc_dec_1/rnn_1/strided_slice_1StridedSlice enc_dec_1/rnn_1/Shape_1:output:0.enc_dec_1/rnn_1/strided_slice_1/stack:output:00enc_dec_1/rnn_1/strided_slice_1/stack_1:output:00enc_dec_1/rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+enc_dec_1/rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
enc_dec_1/rnn_1/TensorArrayV2TensorListReserve4enc_dec_1/rnn_1/TensorArrayV2/element_shape:output:0(enc_dec_1/rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Eenc_dec_1/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7enc_dec_1/rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorenc_dec_1/rnn_1/transpose:y:0Nenc_dec_1/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???o
%enc_dec_1/rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'enc_dec_1/rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'enc_dec_1/rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
enc_dec_1/rnn_1/strided_slice_2StridedSliceenc_dec_1/rnn_1/transpose:y:0.enc_dec_1/rnn_1/strided_slice_2/stack:output:00enc_dec_1/rnn_1/strided_slice_2/stack_1:output:00enc_dec_1/rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
Eenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpNenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
6enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul(enc_dec_1/rnn_1/strided_slice_2:output:0Menc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Genc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpPenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
8enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulenc_dec_1/rnn_1/zeros:output:0Oenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
3enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/addAddV2@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:0Benc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
Fenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpOenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
7enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd7enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/add:z:0Nenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
?enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
5enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/splitSplitHenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:0@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
7enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid>enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
9enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid>enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
3enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/mulMul=enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0 enc_dec_1/rnn_1/zeros_1:output:0*
T0*(
_output_shapes
:???????????
4enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/TanhTanh>enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
5enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul;enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:08enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
5enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/add_1AddV27enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul:z:09enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
9enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid>enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
6enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh9enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
5enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul=enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:0:enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????~
-enc_dec_1/rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
enc_dec_1/rnn_1/TensorArrayV2_1TensorListReserve6enc_dec_1/rnn_1/TensorArrayV2_1/element_shape:output:0(enc_dec_1/rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???V
enc_dec_1/rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(enc_dec_1/rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
"enc_dec_1/rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
enc_dec_1/rnn_1/whileWhile+enc_dec_1/rnn_1/while/loop_counter:output:01enc_dec_1/rnn_1/while/maximum_iterations:output:0enc_dec_1/rnn_1/time:output:0(enc_dec_1/rnn_1/TensorArrayV2_1:handle:0enc_dec_1/rnn_1/zeros:output:0 enc_dec_1/rnn_1/zeros_1:output:0(enc_dec_1/rnn_1/strided_slice_1:output:0Genc_dec_1/rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Nenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourcePenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceOenc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *.
body&R$
"enc_dec_1_rnn_1_while_body_1493590*.
cond&R$
"enc_dec_1_rnn_1_while_cond_1493589*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
@enc_dec_1/rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
2enc_dec_1/rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackenc_dec_1/rnn_1/while:output:3Ienc_dec_1/rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:?	??????????*
element_dtype0x
%enc_dec_1/rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????q
'enc_dec_1/rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'enc_dec_1/rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
enc_dec_1/rnn_1/strided_slice_3StridedSlice;enc_dec_1/rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0.enc_dec_1/rnn_1/strided_slice_3/stack:output:00enc_dec_1/rnn_1/strided_slice_3/stack_1:output:00enc_dec_1/rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_masku
 enc_dec_1/rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
enc_dec_1/rnn_1/transpose_1	Transpose;enc_dec_1/rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0)enc_dec_1/rnn_1/transpose_1/perm:output:0*
T0*-
_output_shapes
:??????????	?r
enc_dec_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    t
enc_dec_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           t
enc_dec_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
enc_dec_1/strided_sliceStridedSliceinput_1&enc_dec_1/strided_slice/stack:output:0(enc_dec_1/strided_slice/stack_1:output:0(enc_dec_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask`
enc_dec_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
enc_dec_1/concatConcatV2 enc_dec_1/strided_slice:output:0input_2enc_dec_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
+enc_dec_1/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp4enc_dec_1_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
enc_dec_1/lstm_cell_3/MatMulMatMulenc_dec_1/concat:output:03enc_dec_1/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-enc_dec_1/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp6enc_dec_1_lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
enc_dec_1/lstm_cell_3/MatMul_1MatMulenc_dec_1/rnn_1/while:output:45enc_dec_1/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
enc_dec_1/lstm_cell_3/addAddV2&enc_dec_1/lstm_cell_3/MatMul:product:0(enc_dec_1/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
,enc_dec_1/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp5enc_dec_1_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
enc_dec_1/lstm_cell_3/BiasAddBiasAddenc_dec_1/lstm_cell_3/add:z:04enc_dec_1/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
%enc_dec_1/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
enc_dec_1/lstm_cell_3/splitSplit.enc_dec_1/lstm_cell_3/split/split_dim:output:0&enc_dec_1/lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
enc_dec_1/lstm_cell_3/SigmoidSigmoid$enc_dec_1/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
enc_dec_1/lstm_cell_3/Sigmoid_1Sigmoid$enc_dec_1/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
enc_dec_1/lstm_cell_3/mulMul#enc_dec_1/lstm_cell_3/Sigmoid_1:y:0enc_dec_1/rnn_1/while:output:5*
T0*(
_output_shapes
:??????????{
enc_dec_1/lstm_cell_3/TanhTanh$enc_dec_1/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
enc_dec_1/lstm_cell_3/mul_1Mul!enc_dec_1/lstm_cell_3/Sigmoid:y:0enc_dec_1/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
enc_dec_1/lstm_cell_3/add_1AddV2enc_dec_1/lstm_cell_3/mul:z:0enc_dec_1/lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:???????????
enc_dec_1/lstm_cell_3/Sigmoid_2Sigmoid$enc_dec_1/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????x
enc_dec_1/lstm_cell_3/Tanh_1Tanhenc_dec_1/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:???????????
enc_dec_1/lstm_cell_3/mul_2Mul#enc_dec_1/lstm_cell_3/Sigmoid_2:y:0 enc_dec_1/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
'enc_dec_1/dense_1/MatMul/ReadVariableOpReadVariableOp0enc_dec_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
enc_dec_1/dense_1/MatMulMatMulenc_dec_1/lstm_cell_3/mul_2:z:0/enc_dec_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(enc_dec_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp1enc_dec_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
enc_dec_1/dense_1/BiasAddBiasAdd"enc_dec_1/dense_1/MatMul:product:00enc_dec_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
enc_dec_1/stackPack"enc_dec_1/dense_1/BiasAdd:output:0*
N*
T0*+
_output_shapes
:?????????m
enc_dec_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
enc_dec_1/transpose	Transposeenc_dec_1/stack:output:0!enc_dec_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????j
IdentityIdentityenc_dec_1/transpose:y:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp)^enc_dec_1/dense_1/BiasAdd/ReadVariableOp(^enc_dec_1/dense_1/MatMul/ReadVariableOp-^enc_dec_1/lstm_cell_3/BiasAdd/ReadVariableOp,^enc_dec_1/lstm_cell_3/MatMul/ReadVariableOp.^enc_dec_1/lstm_cell_3/MatMul_1/ReadVariableOpG^enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpF^enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpH^enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp^enc_dec_1/rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 2T
(enc_dec_1/dense_1/BiasAdd/ReadVariableOp(enc_dec_1/dense_1/BiasAdd/ReadVariableOp2R
'enc_dec_1/dense_1/MatMul/ReadVariableOp'enc_dec_1/dense_1/MatMul/ReadVariableOp2\
,enc_dec_1/lstm_cell_3/BiasAdd/ReadVariableOp,enc_dec_1/lstm_cell_3/BiasAdd/ReadVariableOp2Z
+enc_dec_1/lstm_cell_3/MatMul/ReadVariableOp+enc_dec_1/lstm_cell_3/MatMul/ReadVariableOp2^
-enc_dec_1/lstm_cell_3/MatMul_1/ReadVariableOp-enc_dec_1/lstm_cell_3/MatMul_1/ReadVariableOp2?
Fenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpFenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2?
Eenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpEenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2?
Genc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpGenc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp2.
enc_dec_1/rnn_1/whileenc_dec_1/rnn_1/while:U Q
,
_output_shapes
:??????????	
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?T
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495787

inputsQ
>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?T
@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??N
?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
&stacked_rnn_cells_1/lstm_cell_2/MatMulMatMulstrided_slice_2:output:0=stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulzeros:output:0?stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/addAddV20stacked_rnn_cells_1/lstm_cell_2/MatMul:product:02stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd'stacked_rnn_cells_1/lstm_cell_2/add:z:0>stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
%stacked_rnn_cells_1/lstm_cell_2/splitSplit8stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:00stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
'stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/mulMul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
$stacked_rnn_cells_1/lstm_cell_2/TanhTanh.stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_1Mul+stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0(stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/add_1AddV2'stacked_rnn_cells_1/lstm_cell_2/mul:z:0)stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
&stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh)stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_2Mul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:0*stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1495702*
condR
while_cond_1495701*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:?	??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:??????????	?h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp7^stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6^stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp8^stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????	: : : 2p
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2n
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2r
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs
?:
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494264

inputs.
stacked_rnn_cells_1_1494181:	?/
stacked_rnn_cells_1_1494183:
??*
stacked_rnn_cells_1_1494185:	?
identity

identity_1

identity_2??+stacked_rnn_cells_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_1_1494181stacked_rnn_cells_1_1494183stacked_rnn_cells_1_1494185*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493797n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_1_1494181stacked_rnn_cells_1_1494183stacked_rnn_cells_1_1494185*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1494194*
condR
while_cond_1494193*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:?	??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:??????????	?h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????|
NoOpNoOp,^stacked_rnn_cells_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????	: : : 2Z
+stacked_rnn_cells_1/StatefulPartitionedCall+stacked_rnn_cells_1/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1496114

inputs
states_0
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1493952

inputs

states
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?B
?	
while_body_1495414
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Y
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0:	?\
Hwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0:
??V
Gwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorW
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?Z
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??T
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	???<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
,while/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Cwhile/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
.while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulwhile_placeholder_2Ewhile/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/addAddV26while/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:08while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
-while/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd-while/stacked_rnn_cells_1/lstm_cell_2/add:z:0Dwhile/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
5while/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+while/stacked_rnn_cells_1/lstm_cell_2/splitSplit>while/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:06while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
-while/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/mulMul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:???????????
*while/stacked_rnn_cells_1/lstm_cell_2/TanhTanh4while/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul1while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0.while/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/add_1AddV2-while/stacked_rnn_cells_1/lstm_cell_2/mul:z:0/while/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
,while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:00while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp=^while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<^while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp>^while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"?
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resourceGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0"?
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2|
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2z
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2~
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1493784

inputs

states
states_11
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????O
TanhTanhsplit:output:2*
T0*(
_output_shapes
:??????????V
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:??????????Z
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?I
?
rnn_1_while_body_1495075(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3'
#rnn_1_while_rnn_1_strided_slice_1_0c
_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0_
Lrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0:	?b
Nrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0:
??\
Mrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0:	?
rnn_1_while_identity
rnn_1_while_identity_1
rnn_1_while_identity_2
rnn_1_while_identity_3
rnn_1_while_identity_4
rnn_1_while_identity_5%
!rnn_1_while_rnn_1_strided_slice_1a
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor]
Jrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?`
Lrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??Z
Krnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	???Brnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?Arnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?Crnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0rnn_1_while_placeholderFrnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Arnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpLrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
2rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Irnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Crnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpNrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
4rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulrnn_1_while_placeholder_2Krnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/addAddV2<rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:0>rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
Brnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpMrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
3rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd3rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add:z:0Jrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
;rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/splitSplitDrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:0<rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
3rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
/rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mulMul9rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0rnn_1_while_placeholder_3*
T0*(
_output_shapes
:???????????
0rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/TanhTanh:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul7rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:04rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1AddV23rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul:z:05rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid:rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
2rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
1rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul9rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:06rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
0rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_1_while_placeholder_1rnn_1_while_placeholder5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:???S
rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
rnn_1/while/addAddV2rnn_1_while_placeholderrnn_1/while/add/y:output:0*
T0*
_output_shapes
: U
rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_1/while/add_1AddV2$rnn_1_while_rnn_1_while_loop_counterrnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
rnn_1/while/IdentityIdentityrnn_1/while/add_1:z:0^rnn_1/while/NoOp*
T0*
_output_shapes
: ?
rnn_1/while/Identity_1Identity*rnn_1_while_rnn_1_while_maximum_iterations^rnn_1/while/NoOp*
T0*
_output_shapes
: k
rnn_1/while/Identity_2Identityrnn_1/while/add:z:0^rnn_1/while/NoOp*
T0*
_output_shapes
: ?
rnn_1/while/Identity_3Identity@rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_1/while/NoOp*
T0*
_output_shapes
: ?
rnn_1/while/Identity_4Identity5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0^rnn_1/while/NoOp*
T0*(
_output_shapes
:???????????
rnn_1/while/Identity_5Identity5rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0^rnn_1/while/NoOp*
T0*(
_output_shapes
:???????????
rnn_1/while/NoOpNoOpC^rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpB^rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpD^rnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "5
rnn_1_while_identityrnn_1/while/Identity:output:0"9
rnn_1_while_identity_1rnn_1/while/Identity_1:output:0"9
rnn_1_while_identity_2rnn_1/while/Identity_2:output:0"9
rnn_1_while_identity_3rnn_1/while/Identity_3:output:0"9
rnn_1_while_identity_4rnn_1/while/Identity_4:output:0"9
rnn_1_while_identity_5rnn_1/while/Identity_5:output:0"H
!rnn_1_while_rnn_1_strided_slice_1#rnn_1_while_rnn_1_strided_slice_1_0"?
Krnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resourceMrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0"?
Lrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceNrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Jrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceLrnn_1_while_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0"?
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
Brnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpBrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2?
Arnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpArnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2?
Crnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpCrnn_1/while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1494193
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1494193___redundant_placeholder05
1while_while_cond_1494193___redundant_placeholder15
1while_while_cond_1494193___redundant_placeholder25
1while_while_cond_1494193___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_lstm_cell_3_layer_call_fn_1496065

inputs
states_0
states_1
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494312p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?

?
+__inference_enc_dec_1_layer_call_fn_1494361
input_1
input_2
unknown:	?
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494342s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????	
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?B
?	
while_body_1495702
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Y
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0:	?\
Hwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0:
??V
Gwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorW
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?Z
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??T
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	???<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
,while/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Cwhile/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
.while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulwhile_placeholder_2Ewhile/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/addAddV26while/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:08while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
-while/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd-while/stacked_rnn_cells_1/lstm_cell_2/add:z:0Dwhile/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
5while/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+while/stacked_rnn_cells_1/lstm_cell_2/splitSplit>while/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:06while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
-while/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/mulMul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:???????????
*while/stacked_rnn_cells_1/lstm_cell_2/TanhTanh4while/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul1while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0.while/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/add_1AddV2-while/stacked_rnn_cells_1/lstm_cell_2/mul:z:0/while/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
,while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:00while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp=^while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<^while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp>^while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"?
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resourceGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0"?
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2|
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2z
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2~
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_1495950

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1493810
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1493810___redundant_placeholder05
1while_while_cond_1493810___redundant_placeholder15
1while_while_cond_1493810___redundant_placeholder25
1while_while_cond_1493810___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494722
input_1
input_2 
rnn_1_1494689:	?!
rnn_1_1494691:
??
rnn_1_1494693:	?&
lstm_cell_3_1494704:	?'
lstm_cell_3_1494706:
??"
lstm_cell_3_1494708:	?"
dense_1_1494713:	?
dense_1_1494715:
identity??dense_1/StatefulPartitionedCall?#lstm_cell_3/StatefulPartitionedCall?rnn_1/StatefulPartitionedCall?
rnn_1/StatefulPartitionedCallStatefulPartitionedCallinput_1rnn_1_1494689rnn_1_1494691rnn_1_1494693*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494264h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0input_2concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0&rnn_1/StatefulPartitionedCall:output:1&rnn_1/StatefulPartitionedCall:output:2lstm_cell_3_1494704lstm_cell_3_1494706lstm_cell_3_1494708*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1494312?
dense_1/StatefulPartitionedCallStatefulPartitionedCall,lstm_cell_3/StatefulPartitionedCall:output:0dense_1_1494713dense_1_1494715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1494332v
stackPack(dense_1/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall$^lstm_cell_3/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????	
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
??
?
#__inference__traced_restore_1496390
file_prefix<
)assignvariableop_enc_dec_1_dense_1_kernel:	?7
)assignvariableop_1_enc_dec_1_dense_1_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: \
Iassignvariableop_7_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel:	?g
Sassignvariableop_8_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel:
??V
Gassignvariableop_9_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias:	?C
0assignvariableop_10_enc_dec_1_lstm_cell_3_kernel:	?N
:assignvariableop_11_enc_dec_1_lstm_cell_3_recurrent_kernel:
??=
.assignvariableop_12_enc_dec_1_lstm_cell_3_bias:	?#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: %
assignvariableop_17_total_2: %
assignvariableop_18_count_2: F
3assignvariableop_19_adam_enc_dec_1_dense_1_kernel_m:	??
1assignvariableop_20_adam_enc_dec_1_dense_1_bias_m:d
Qassignvariableop_21_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_m:	?o
[assignvariableop_22_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_m:
??^
Oassignvariableop_23_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_m:	?J
7assignvariableop_24_adam_enc_dec_1_lstm_cell_3_kernel_m:	?U
Aassignvariableop_25_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_m:
??D
5assignvariableop_26_adam_enc_dec_1_lstm_cell_3_bias_m:	?F
3assignvariableop_27_adam_enc_dec_1_dense_1_kernel_v:	??
1assignvariableop_28_adam_enc_dec_1_dense_1_bias_v:d
Qassignvariableop_29_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_v:	?o
[assignvariableop_30_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_v:
??^
Oassignvariableop_31_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_v:	?J
7assignvariableop_32_adam_enc_dec_1_lstm_cell_3_kernel_v:	?U
Aassignvariableop_33_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_v:
??D
5assignvariableop_34_adam_enc_dec_1_lstm_cell_3_bias_v:	?
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp)assignvariableop_enc_dec_1_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_enc_dec_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpIassignvariableop_7_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpSassignvariableop_8_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpGassignvariableop_9_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_enc_dec_1_lstm_cell_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_enc_dec_1_lstm_cell_3_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_enc_dec_1_lstm_cell_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_enc_dec_1_dense_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_enc_dec_1_dense_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpQassignvariableop_21_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp[assignvariableop_22_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpOassignvariableop_23_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp7assignvariableop_24_adam_enc_dec_1_lstm_cell_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpAassignvariableop_25_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_enc_dec_1_lstm_cell_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_enc_dec_1_dense_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_enc_dec_1_dense_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpQassignvariableop_29_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp[assignvariableop_30_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpOassignvariableop_31_adam_enc_dec_1_rnn_1_stacked_rnn_cells_1_lstm_cell_2_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp7assignvariableop_32_adam_enc_dec_1_lstm_cell_3_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpAassignvariableop_33_adam_enc_dec_1_lstm_cell_3_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_enc_dec_1_lstm_cell_3_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
while_cond_1495845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1495845___redundant_placeholder05
1while_while_cond_1495845___redundant_placeholder15
1while_while_cond_1495845___redundant_placeholder25
1while_while_cond_1495845___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1495015
inputs_0
inputs_1W
Drnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?Z
Frnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??T
Ernn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	?=
*lstm_cell_3_matmul_readvariableop_resource:	?@
,lstm_cell_3_matmul_1_readvariableop_resource:
??:
+lstm_cell_3_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?"lstm_cell_3/BiasAdd/ReadVariableOp?!lstm_cell_3/MatMul/ReadVariableOp?#lstm_cell_3/MatMul_1/ReadVariableOp?<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?rnn_1/whileC
rnn_1/ShapeShapeinputs_0*
T0*
_output_shapes
:c
rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_sliceStridedSlicernn_1/Shape:output:0"rnn_1/strided_slice/stack:output:0$rnn_1/strided_slice/stack_1:output:0$rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
rnn_1/zeros/packedPackrnn_1/strided_slice:output:0rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_1/zerosFillrnn_1/zeros/packed:output:0rnn_1/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Y
rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
rnn_1/zeros_1/packedPackrnn_1/strided_slice:output:0rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:X
rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
rnn_1/zeros_1Fillrnn_1/zeros_1/packed:output:0rnn_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????i
rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rnn_1/transpose	Transposeinputs_0rnn_1/transpose/perm:output:0*
T0*,
_output_shapes
:?	?????????P
rnn_1/Shape_1Shapernn_1/transpose:y:0*
T0*
_output_shapes
:e
rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_slice_1StridedSlicernn_1/Shape_1:output:0$rnn_1/strided_slice_1/stack:output:0&rnn_1/strided_slice_1/stack_1:output:0&rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
rnn_1/TensorArrayV2TensorListReserve*rnn_1/TensorArrayV2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
-rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_1/transpose:y:0Drnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???e
rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_slice_2StridedSlicernn_1/transpose:y:0$rnn_1/strided_slice_2/stack:output:0&rnn_1/strided_slice_2/stack_1:output:0&rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpDrnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
,rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMulrnn_1/strided_slice_2:output:0Crnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpFrnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
.rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulrnn_1/zeros:output:0Ernn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)rnn_1/stacked_rnn_cells_1/lstm_cell_2/addAddV26rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:08rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpErnn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
-rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd-rnn_1/stacked_rnn_cells_1/lstm_cell_2/add:z:0Drnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
5rnn_1/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/splitSplit>rnn_1/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:06rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
-rnn_1/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
)rnn_1/stacked_rnn_cells_1/lstm_cell_2/mulMul3rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0rnn_1/zeros_1:output:0*
T0*(
_output_shapes
:???????????
*rnn_1/stacked_rnn_cells_1/lstm_cell_2/TanhTanh4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul1rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0.rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/add_1AddV2-rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul:z:0/rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
/rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid4rnn_1/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
,rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh/rnn_1/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
+rnn_1/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul3rnn_1/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:00rnn_1/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????t
#rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
rnn_1/TensorArrayV2_1TensorListReserve,rnn_1/TensorArrayV2_1/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???L

rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Z
rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
rnn_1/whileWhile!rnn_1/while/loop_counter:output:0'rnn_1/while/maximum_iterations:output:0rnn_1/time:output:0rnn_1/TensorArrayV2_1:handle:0rnn_1/zeros:output:0rnn_1/zeros_1:output:0rnn_1/strided_slice_1:output:0=rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Drnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceFrnn_1_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceErnn_1_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
rnn_1_while_body_1494893*$
condR
rnn_1_while_cond_1494892*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
(rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_1/while:output:3?rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:?	??????????*
element_dtype0n
rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????g
rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rnn_1/strided_slice_3StridedSlice1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_1/strided_slice_3/stack:output:0&rnn_1/strided_slice_3/stack_1:output:0&rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskk
rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
rnn_1/transpose_1	Transpose1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0rnn_1/transpose_1/perm:output:0*
T0*-
_output_shapes
:??????????	?h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
strided_sliceStridedSliceinputs_0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_3/MatMulMatMulconcat:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/MatMul_1MatMulrnn_1/while:output:4+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_splitm
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????o
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????z
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0rnn_1/while:output:5*
T0*(
_output_shapes
:??????????g
lstm_cell_3/TanhTanhlstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????z
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????y
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*(
_output_shapes
:??????????o
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_1/MatMulMatMullstm_cell_3/mul_2:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
stackPackdense_1/BiasAdd:output:0*
N*
T0*+
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp=^rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<^rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp>^rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp^rnn_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:??????????	:?????????: : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2|
<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<rnn_1/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2z
;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp;rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2~
=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp=rnn_1/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp2
rnn_1/whilernn_1/while:V R
,
_output_shapes
:??????????	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?B
?	
while_body_1495558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Y
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0:	?\
Hwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0:
??V
Gwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorW
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?Z
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??T
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	???<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
,while/stacked_rnn_cells_1/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Cwhile/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
.while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulwhile_placeholder_2Ewhile/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/addAddV26while/stacked_rnn_cells_1/lstm_cell_2/MatMul:product:08while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOpGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
-while/stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd-while/stacked_rnn_cells_1/lstm_cell_2/add:z:0Dwhile/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
5while/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+while/stacked_rnn_cells_1/lstm_cell_2/splitSplit>while/stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:06while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
-while/stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
)while/stacked_rnn_cells_1/lstm_cell_2/mulMul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:???????????
*while/stacked_rnn_cells_1/lstm_cell_2/TanhTanh4while/stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_1Mul1while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0.while/stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/add_1AddV2-while/stacked_rnn_cells_1/lstm_cell_2/mul:z:0/while/stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
/while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid4while/stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
,while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
+while/stacked_rnn_cells_1/lstm_cell_2/mul_2Mul3while/stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:00while/stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity/while/stacked_rnn_cells_1/lstm_cell_2/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity/while/stacked_rnn_cells_1/lstm_cell_2/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp=^while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<^while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp>^while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"?
Ewhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resourceGwhile_stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource_0"?
Fwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resourceHwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource_0"?
Dwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resourceFwhile_stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2|
<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp<while/stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2z
;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp;while/stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2~
=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp=while/stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1494498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1494498___redundant_placeholder05
1while_while_cond_1494498___redundant_placeholder15
1while_while_cond_1494498___redundant_placeholder25
1while_while_cond_1494498___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_rnn_1_layer_call_fn_1495325
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494129p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?T
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495931

inputsQ
>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource:	?T
@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource:
??N
?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp?5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp?7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:?	?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOpReadVariableOp>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
&stacked_rnn_cells_1/lstm_cell_2/MatMulMatMulstrided_slice_2:output:0=stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOpReadVariableOp@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(stacked_rnn_cells_1/lstm_cell_2/MatMul_1MatMulzeros:output:0?stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/addAddV20stacked_rnn_cells_1/lstm_cell_2/MatMul:product:02stacked_rnn_cells_1/lstm_cell_2/MatMul_1:product:0*
T0*(
_output_shapes
:???????????
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOpReadVariableOp?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'stacked_rnn_cells_1/lstm_cell_2/BiasAddBiasAdd'stacked_rnn_cells_1/lstm_cell_2/add:z:0>stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
/stacked_rnn_cells_1/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
%stacked_rnn_cells_1/lstm_cell_2/splitSplit8stacked_rnn_cells_1/lstm_cell_2/split/split_dim:output:00stacked_rnn_cells_1/lstm_cell_2/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split?
'stacked_rnn_cells_1/lstm_cell_2/SigmoidSigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
#stacked_rnn_cells_1/lstm_cell_2/mulMul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
$stacked_rnn_cells_1/lstm_cell_2/TanhTanh.stacked_rnn_cells_1/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_1Mul+stacked_rnn_cells_1/lstm_cell_2/Sigmoid:y:0(stacked_rnn_cells_1/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/add_1AddV2'stacked_rnn_cells_1/lstm_cell_2/mul:z:0)stacked_rnn_cells_1/lstm_cell_2/mul_1:z:0*
T0*(
_output_shapes
:???????????
)stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2Sigmoid.stacked_rnn_cells_1/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:???????????
&stacked_rnn_cells_1/lstm_cell_2/Tanh_1Tanh)stacked_rnn_cells_1/lstm_cell_2/add_1:z:0*
T0*(
_output_shapes
:???????????
%stacked_rnn_cells_1/lstm_cell_2/mul_2Mul-stacked_rnn_cells_1/lstm_cell_2/Sigmoid_2:y:0*stacked_rnn_cells_1/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0>stacked_rnn_cells_1_lstm_cell_2_matmul_readvariableop_resource@stacked_rnn_cells_1_lstm_cell_2_matmul_1_readvariableop_resource?stacked_rnn_cells_1_lstm_cell_2_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1495846*
condR
while_cond_1495845*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:?	??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:??????????	?h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp7^stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6^stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp8^stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????	: : : 2p
6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp6stacked_rnn_cells_1/lstm_cell_2/BiasAdd/ReadVariableOp2n
5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp5stacked_rnn_cells_1/lstm_cell_2/MatMul/ReadVariableOp2r
7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp7stacked_rnn_cells_1/lstm_cell_2/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????	
 
_user_specified_nameinputs
?:
?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1494129

inputs.
stacked_rnn_cells_1_1494046:	?/
stacked_rnn_cells_1_1494048:
??*
stacked_rnn_cells_1_1494050:	?
identity

identity_1

identity_2??+stacked_rnn_cells_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+stacked_rnn_cells_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_1_1494046stacked_rnn_cells_1_1494048stacked_rnn_cells_1_1494050*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1493965n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_1_1494046stacked_rnn_cells_1_1494048stacked_rnn_cells_1_1494050*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1494059*
condR
while_cond_1494058*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:??????????`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:??????????|
NoOpNoOp,^stacked_rnn_cells_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+stacked_rnn_cells_1/StatefulPartitionedCall+stacked_rnn_cells_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_1495413
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1495413___redundant_placeholder05
1while_while_cond_1495413___redundant_placeholder15
1while_while_cond_1495413___redundant_placeholder25
1while_while_cond_1495413___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_1494332

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????	
;
input_20
serving_default_input_2:0?????????@
output_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	enc_units
	dec_units
encoder_cells
encoder_stacked
encoder_rnn
decoder_cells
	dense
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
v__call__
*w&call_and_return_all_conditional_losses
x_default_save_signature"
_tf_keras_model
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
	cells
	variables
trainable_variables
regularization_losses
	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
'
0"
trackable_list_wrapper
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemfmg$mh%mi&mj'mk(ml)mmvnvo$vp%vq&vr'vs(vt)vu"
	optimizer
X
$0
%1
&2
'3
(4
)5
6
7"
trackable_list_wrapper
X
$0
%1
&2
'3
(4
)5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
		variables

trainable_variables
regularization_losses
v__call__
x_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
?
/
state_size

$kernel
%recurrent_kernel
&bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
'
90"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
?
@
state_size

'kernel
(recurrent_kernel
)bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
+:)	?2enc_dec_1/dense_1/kernel
$:"2enc_dec_1/dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
I:G	?26enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel
T:R
??2@enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel
C:A?24enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias
/:-	?2enc_dec_1/lstm_cell_3/kernel
::8
??2&enc_dec_1/lstm_cell_3/recurrent_kernel
):'?2enc_dec_1/lstm_cell_3/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
0	variables
1trainable_variables
2regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Xtotal
	Ycount
Z	variables
[	keras_api"
_tf_keras_metric
^
	\total
	]count
^
_fn_kwargs
_	variables
`	keras_api"
_tf_keras_metric
^
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
X0
Y1"
trackable_list_wrapper
-
Z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
0:.	?2Adam/enc_dec_1/dense_1/kernel/m
):'2Adam/enc_dec_1/dense_1/bias/m
N:L	?2=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/m
Y:W
??2GAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/m
H:F?2;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/m
4:2	?2#Adam/enc_dec_1/lstm_cell_3/kernel/m
?:=
??2-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/m
.:,?2!Adam/enc_dec_1/lstm_cell_3/bias/m
0:.	?2Adam/enc_dec_1/dense_1/kernel/v
):'2Adam/enc_dec_1/dense_1/bias/v
N:L	?2=Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/kernel/v
Y:W
??2GAdam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/recurrent_kernel/v
H:F?2;Adam/enc_dec_1/rnn_1/stacked_rnn_cells_1/lstm_cell_2/bias/v
4:2	?2#Adam/enc_dec_1/lstm_cell_3/kernel/v
?:=
??2-Adam/enc_dec_1/lstm_cell_3/recurrent_kernel/v
.:,?2!Adam/enc_dec_1/lstm_cell_3/bias/v
?2?
+__inference_enc_dec_1_layer_call_fn_1494361
+__inference_enc_dec_1_layer_call_fn_1494811
+__inference_enc_dec_1_layer_call_fn_1494833
+__inference_enc_dec_1_layer_call_fn_1494685?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1495015
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1495197
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494722
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494759?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1493712input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stacked_rnn_cells_1_layer_call_fn_1495214
5__inference_stacked_rnn_cells_1_layer_call_fn_1495231?
???
FullArgSpec@
args8?5
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1495263
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1495295?
???
FullArgSpec@
args8?5
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_rnn_1_layer_call_fn_1495310
'__inference_rnn_1_layer_call_fn_1495325
'__inference_rnn_1_layer_call_fn_1495340
'__inference_rnn_1_layer_call_fn_1495355?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495499
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495643
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495787
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495931?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_1495940?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_1495950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1494789input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_lstm_cell_2_layer_call_fn_1495967
-__inference_lstm_cell_2_layer_call_fn_1495984?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1496016
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1496048?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_lstm_cell_3_layer_call_fn_1496065
-__inference_lstm_cell_3_layer_call_fn_1496082?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1496114
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1496146?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
"__inference__wrapped_model_1493712?$%&'()]?Z
S?P
N?K
&?#
input_1??????????	
!?
input_2?????????
? "7?4
2
output_1&?#
output_1??????????
D__inference_dense_1_layer_call_and_return_conditional_losses_1495950]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_1_layer_call_fn_1495940P0?-
&?#
!?
inputs??????????
? "???????????
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494722?$%&'()a?^
W?T
N?K
&?#
input_1??????????	
!?
input_2?????????
p 
? ")?&
?
0?????????
? ?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1494759?$%&'()a?^
W?T
N?K
&?#
input_1??????????	
!?
input_2?????????
p
? ")?&
?
0?????????
? ?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1495015?$%&'()c?`
Y?V
P?M
'?$
inputs/0??????????	
"?
inputs/1?????????
p 
? ")?&
?
0?????????
? ?
F__inference_enc_dec_1_layer_call_and_return_conditional_losses_1495197?$%&'()c?`
Y?V
P?M
'?$
inputs/0??????????	
"?
inputs/1?????????
p
? ")?&
?
0?????????
? ?
+__inference_enc_dec_1_layer_call_fn_1494361?$%&'()a?^
W?T
N?K
&?#
input_1??????????	
!?
input_2?????????
p 
? "???????????
+__inference_enc_dec_1_layer_call_fn_1494685?$%&'()a?^
W?T
N?K
&?#
input_1??????????	
!?
input_2?????????
p
? "???????????
+__inference_enc_dec_1_layer_call_fn_1494811?$%&'()c?`
Y?V
P?M
'?$
inputs/0??????????	
"?
inputs/1?????????
p 
? "???????????
+__inference_enc_dec_1_layer_call_fn_1494833?$%&'()c?`
Y?V
P?M
'?$
inputs/0??????????	
"?
inputs/1?????????
p
? "???????????
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1496016?$%&??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
H__inference_lstm_cell_2_layer_call_and_return_conditional_losses_1496048?$%&??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
-__inference_lstm_cell_2_layer_call_fn_1495967?$%&??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
-__inference_lstm_cell_2_layer_call_fn_1495984?$%&??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1496114?'()??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
H__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1496146?'()??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
-__inference_lstm_cell_3_layer_call_fn_1496065?'()??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
-__inference_lstm_cell_3_layer_call_fn_1496082?'()??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495499?$%&S?P
I?F
4?1
/?,
inputs/0??????????????????

 
p 

 

 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495643?$%&S?P
I?F
4?1
/?,
inputs/0??????????????????

 
p

 

 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495787?$%&D?A
:?7
%?"
inputs??????????	

 
p 

 

 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
B__inference_rnn_1_layer_call_and_return_conditional_losses_1495931?$%&D?A
:?7
%?"
inputs??????????	

 
p

 

 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
'__inference_rnn_1_layer_call_fn_1495310?$%&S?P
I?F
4?1
/?,
inputs/0??????????????????

 
p 

 

 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
'__inference_rnn_1_layer_call_fn_1495325?$%&S?P
I?F
4?1
/?,
inputs/0??????????????????

 
p

 

 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
'__inference_rnn_1_layer_call_fn_1495340?$%&D?A
:?7
%?"
inputs??????????	

 
p 

 

 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
'__inference_rnn_1_layer_call_fn_1495355?$%&D?A
:?7
%?"
inputs??????????	

 
p

 

 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
%__inference_signature_wrapper_1494789?$%&'()n?k
? 
d?a
1
input_1&?#
input_1??????????	
,
input_2!?
input_2?????????"7?4
2
output_1&?#
output_1??????????
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1495263?$%&???
???
 ?
inputs?????????
V?S
Q?N
%?"

states/0/0??????????
%?"

states/0/1??????????

 
p 
? "?|
u?r
?
0/0??????????
P?M
K?H
"?
0/1/0/0??????????
"?
0/1/0/1??????????
? ?
P__inference_stacked_rnn_cells_1_layer_call_and_return_conditional_losses_1495295?$%&???
???
 ?
inputs?????????
V?S
Q?N
%?"

states/0/0??????????
%?"

states/0/1??????????

 
p
? "?|
u?r
?
0/0??????????
P?M
K?H
"?
0/1/0/0??????????
"?
0/1/0/1??????????
? ?
5__inference_stacked_rnn_cells_1_layer_call_fn_1495214?$%&???
???
 ?
inputs?????????
V?S
Q?N
%?"

states/0/0??????????
%?"

states/0/1??????????

 
p 
? "o?l
?
0??????????
L?I
G?D
 ?
1/0/0??????????
 ?
1/0/1???????????
5__inference_stacked_rnn_cells_1_layer_call_fn_1495231?$%&???
???
 ?
inputs?????????
V?S
Q?N
%?"

states/0/0??????????
%?"

states/0/1??????????

 
p
? "o?l
?
0??????????
L?I
G?D
 ?
1/0/0??????????
 ?
1/0/1??????????