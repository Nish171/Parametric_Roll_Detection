 
Ã
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¸Ö

enc_dec_12/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameenc_dec_12/dense_14/kernel

.enc_dec_12/dense_14/kernel/Read/ReadVariableOpReadVariableOpenc_dec_12/dense_14/kernel*
_output_shapes
:	*
dtype0

enc_dec_12/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameenc_dec_12/dense_14/bias

,enc_dec_12/dense_14/bias/Read/ReadVariableOpReadVariableOpenc_dec_12/dense_14/bias*
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
Ñ
:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*K
shared_name<:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel
Ê
Nenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/Read/ReadVariableOpReadVariableOp:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel*
_output_shapes
:	*
dtype0
æ
Denc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*U
shared_nameFDenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel
ß
Xenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/Read/ReadVariableOpReadVariableOpDenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel* 
_output_shapes
:
*
dtype0
É
8enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias
Â
Lenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/Read/ReadVariableOpReadVariableOp8enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias*
_output_shapes	
:*
dtype0

enc_dec_12/lstm_cell_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name enc_dec_12/lstm_cell_41/kernel

2enc_dec_12/lstm_cell_41/kernel/Read/ReadVariableOpReadVariableOpenc_dec_12/lstm_cell_41/kernel*
_output_shapes
:	*
dtype0
®
(enc_dec_12/lstm_cell_41/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(enc_dec_12/lstm_cell_41/recurrent_kernel
§
<enc_dec_12/lstm_cell_41/recurrent_kernel/Read/ReadVariableOpReadVariableOp(enc_dec_12/lstm_cell_41/recurrent_kernel* 
_output_shapes
:
*
dtype0

enc_dec_12/lstm_cell_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameenc_dec_12/lstm_cell_41/bias

0enc_dec_12/lstm_cell_41/bias/Read/ReadVariableOpReadVariableOpenc_dec_12/lstm_cell_41/bias*
_output_shapes	
:*
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

!Adam/enc_dec_12/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/enc_dec_12/dense_14/kernel/m

5Adam/enc_dec_12/dense_14/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/enc_dec_12/dense_14/kernel/m*
_output_shapes
:	*
dtype0

Adam/enc_dec_12/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/enc_dec_12/dense_14/bias/m

3Adam/enc_dec_12/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_dec_12/dense_14/bias/m*
_output_shapes
:*
dtype0
ß
AAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*R
shared_nameCAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/m
Ø
UAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/m*
_output_shapes
:	*
dtype0
ô
KAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*\
shared_nameMKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/m
í
_Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/m* 
_output_shapes
:
*
dtype0
×
?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/m
Ð
SAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/m/Read/ReadVariableOpReadVariableOp?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/m*
_output_shapes	
:*
dtype0
§
%Adam/enc_dec_12/lstm_cell_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%Adam/enc_dec_12/lstm_cell_41/kernel/m
 
9Adam/enc_dec_12/lstm_cell_41/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/enc_dec_12/lstm_cell_41/kernel/m*
_output_shapes
:	*
dtype0
¼
/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/m
µ
CAdam/enc_dec_12/lstm_cell_41/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

#Adam/enc_dec_12/lstm_cell_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/enc_dec_12/lstm_cell_41/bias/m

7Adam/enc_dec_12/lstm_cell_41/bias/m/Read/ReadVariableOpReadVariableOp#Adam/enc_dec_12/lstm_cell_41/bias/m*
_output_shapes	
:*
dtype0

!Adam/enc_dec_12/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/enc_dec_12/dense_14/kernel/v

5Adam/enc_dec_12/dense_14/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/enc_dec_12/dense_14/kernel/v*
_output_shapes
:	*
dtype0

Adam/enc_dec_12/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/enc_dec_12/dense_14/bias/v

3Adam/enc_dec_12/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_dec_12/dense_14/bias/v*
_output_shapes
:*
dtype0
ß
AAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*R
shared_nameCAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/v
Ø
UAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/v*
_output_shapes
:	*
dtype0
ô
KAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*\
shared_nameMKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/v
í
_Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/v* 
_output_shapes
:
*
dtype0
×
?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/v
Ð
SAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/v/Read/ReadVariableOpReadVariableOp?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/v*
_output_shapes	
:*
dtype0
§
%Adam/enc_dec_12/lstm_cell_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%Adam/enc_dec_12/lstm_cell_41/kernel/v
 
9Adam/enc_dec_12/lstm_cell_41/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/enc_dec_12/lstm_cell_41/kernel/v*
_output_shapes
:	*
dtype0
¼
/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/v
µ
CAdam/enc_dec_12/lstm_cell_41/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

#Adam/enc_dec_12/lstm_cell_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/enc_dec_12/lstm_cell_41/bias/v

7Adam/enc_dec_12/lstm_cell_41/bias/v/Read/ReadVariableOpReadVariableOp#Adam/enc_dec_12/lstm_cell_41/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
¬B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*çA
valueÝABÚA BÓA
Á
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
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 

0*

	cells
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
ª
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

0*
¦

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
Ú
(iter

)beta_1

*beta_2
	+decay
,learning_rate mv!mw-mx.my/mz0m{1m|2m} v~!v-v.v/v0v1v2v*
<
-0
.1
/2
03
14
25
 6
!7*
<
-0
.1
/2
03
14
25
 6
!7*
* 
°
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

8serving_default* 
ã
9
state_size

-kernel
.recurrent_kernel
/bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses*

-0
.1
/2*

-0
.1
/2*
* 

Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
	
F0* 

-0
.1
/2*

-0
.1
/2*
* 


Gstates
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ã
M
state_size

0kernel
1recurrent_kernel
2bias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R_random_generator
S__call__
*T&call_and_return_all_conditional_losses*
[U
VARIABLE_VALUEenc_dec_12/dense_14/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEenc_dec_12/dense_14/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 

Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEDenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEenc_dec_12/lstm_cell_41/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(enc_dec_12/lstm_cell_41/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEenc_dec_12/lstm_cell_41/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

Z0
[1
\2*
* 
* 
* 
* 

-0
.1
/2*

-0
.1
/2*
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
* 
* 

0*
* 
* 
* 
* 
	
b0* 
* 

0*
* 
* 
* 
* 

00
11
22*

00
11
22*
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
8
	htotal
	icount
j	variables
k	keras_api*
H
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api*
H
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

j	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

l0
m1*

o	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

q0
r1*

t	variables*
~x
VARIABLE_VALUE!Adam/enc_dec_12/dense_14/kernel/mCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/enc_dec_12/dense_14/bias/mAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¨¡
VARIABLE_VALUEKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/enc_dec_12/lstm_cell_41/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/enc_dec_12/lstm_cell_41/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/enc_dec_12/dense_14/kernel/vCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/enc_dec_12/dense_14/bias/vAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¨¡
VARIABLE_VALUEKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/enc_dec_12/lstm_cell_41/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/enc_dec_12/lstm_cell_41/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	*
dtype0*!
shape:ÿÿÿÿÿÿÿÿÿ°	
¤
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernelDenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel8enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/biasenc_dec_12/lstm_cell_41/kernel(enc_dec_12/lstm_cell_41/recurrent_kernelenc_dec_12/lstm_cell_41/biasenc_dec_12/dense_14/kernelenc_dec_12/dense_14/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1331427
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ó
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.enc_dec_12/dense_14/kernel/Read/ReadVariableOp,enc_dec_12/dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpNenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/Read/ReadVariableOpXenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/Read/ReadVariableOpLenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/Read/ReadVariableOp2enc_dec_12/lstm_cell_41/kernel/Read/ReadVariableOp<enc_dec_12/lstm_cell_41/recurrent_kernel/Read/ReadVariableOp0enc_dec_12/lstm_cell_41/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp5Adam/enc_dec_12/dense_14/kernel/m/Read/ReadVariableOp3Adam/enc_dec_12/dense_14/bias/m/Read/ReadVariableOpUAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/m/Read/ReadVariableOp_Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/m/Read/ReadVariableOpSAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/m/Read/ReadVariableOp9Adam/enc_dec_12/lstm_cell_41/kernel/m/Read/ReadVariableOpCAdam/enc_dec_12/lstm_cell_41/recurrent_kernel/m/Read/ReadVariableOp7Adam/enc_dec_12/lstm_cell_41/bias/m/Read/ReadVariableOp5Adam/enc_dec_12/dense_14/kernel/v/Read/ReadVariableOp3Adam/enc_dec_12/dense_14/bias/v/Read/ReadVariableOpUAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/v/Read/ReadVariableOp_Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/v/Read/ReadVariableOpSAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/v/Read/ReadVariableOp9Adam/enc_dec_12/lstm_cell_41/kernel/v/Read/ReadVariableOpCAdam/enc_dec_12/lstm_cell_41/recurrent_kernel/v/Read/ReadVariableOp7Adam/enc_dec_12/lstm_cell_41/bias/v/Read/ReadVariableOpConst*0
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_1332504
²
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_dec_12/dense_14/kernelenc_dec_12/dense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernelDenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel8enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/biasenc_dec_12/lstm_cell_41/kernel(enc_dec_12/lstm_cell_41/recurrent_kernelenc_dec_12/lstm_cell_41/biastotalcounttotal_1count_1total_2count_2!Adam/enc_dec_12/dense_14/kernel/mAdam/enc_dec_12/dense_14/bias/mAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/mKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/m?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/m%Adam/enc_dec_12/lstm_cell_41/kernel/m/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/m#Adam/enc_dec_12/lstm_cell_41/bias/m!Adam/enc_dec_12/dense_14/kernel/vAdam/enc_dec_12/dense_14/bias/vAAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/vKAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/v?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/v%Adam/enc_dec_12/lstm_cell_41/kernel/v/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/v#Adam/enc_dec_12/lstm_cell_41/bias/v*/
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1332619Ý
áU
º
C__inference_rnn_12_layer_call_and_return_conditional_losses_1332017

inputsS
@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	V
Bstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
P
Astacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp¢while;
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¹
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0À
(stacked_rnn_cells_12/lstm_cell_40/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0º
*stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulzeros:output:0Astacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%stacked_rnn_cells_12/lstm_cell_40/addAddV22stacked_rnn_cells_12/lstm_cell_40/MatMul:product:04stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
)stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd)stacked_rnn_cells_12/lstm_cell_40/add:z:0@stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'stacked_rnn_cells_12/lstm_cell_40/splitSplit:stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:02stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
%stacked_rnn_cells_12/lstm_cell_40/mulMul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stacked_rnn_cells_12/lstm_cell_40/TanhTanh0stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
'stacked_rnn_cells_12/lstm_cell_40/mul_1Mul-stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:0*stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'stacked_rnn_cells_12/lstm_cell_40/add_1AddV2)stacked_rnn_cells_12/lstm_cell_40/mul:z:0+stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh+stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
'stacked_rnn_cells_12/lstm_cell_40/mul_2Mul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:0,stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ç
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1331932*
condR
while_cond_1331931*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp9^stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8^stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:^stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ°	: : : 2t
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2v
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
Ê:
É
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330517

inputs/
stacked_rnn_cells_12_1330434:	0
stacked_rnn_cells_12_1330436:
+
stacked_rnn_cells_12_1330438:	
identity

identity_1

identity_2¢,stacked_rnn_cells_12/StatefulPartitionedCall¢while;
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask§
,stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_12_1330434stacked_rnn_cells_12_1330436stacked_rnn_cells_12_1330438*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330052n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_12_1330434stacked_rnn_cells_12_1330436stacked_rnn_cells_12_1330438*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1330447*
condR
while_cond_1330446*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
NoOpNoOp-^stacked_rnn_cells_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ°	: : : 2\
,stacked_rnn_cells_12/StatefulPartitionedCall,stacked_rnn_cells_12/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
Ó
ô
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330890

inputs!
rnn_12_1330859:	"
rnn_12_1330861:

rnn_12_1330863:	'
lstm_cell_41_1330872:	(
lstm_cell_41_1330874:
#
lstm_cell_41_1330876:	#
dense_14_1330881:	
dense_14_1330883:
identity¢ dense_14/StatefulPartitionedCall¢$lstm_cell_41/StatefulPartitionedCall¢rnn_12/StatefulPartitionedCall«
rnn_12/StatefulPartitionedCallStatefulPartitionedCallinputsrnn_12_1330859rnn_12_1330861rnn_12_1330863*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330820h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ü
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask­
$lstm_cell_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0'rnn_12/StatefulPartitionedCall:output:1'rnn_12/StatefulPartitionedCall:output:2lstm_cell_41_1330872lstm_cell_41_1330874lstm_cell_41_1330876*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330676
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-lstm_cell_41/StatefulPartitionedCall:output:0dense_14_1330881dense_14_1330883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1330583w
stackPack)dense_14/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall%^lstm_cell_41/StatefulPartitionedCall^rnn_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$lstm_cell_41/StatefulPartitionedCall$lstm_cell_41/StatefulPartitionedCall2@
rnn_12/StatefulPartitionedCallrnn_12/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
¾
È
while_cond_1330313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1330313___redundant_placeholder05
1while_while_cond_1330313___redundant_placeholder15
1while_while_cond_1330313___redundant_placeholder25
1while_while_cond_1330313___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

Ø
(__inference_rnn_12_layer_call_fn_1331555
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330384p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


6__inference_stacked_rnn_cells_12_layer_call_fn_1331461

inputs

states_0_0

states_0_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall»
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330220p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1
à

I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1332278

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ä
É
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330220

inputs

states
states_1'
lstm_cell_40_1330208:	(
lstm_cell_40_1330210:
#
lstm_cell_40_1330212:	
identity

identity_1

identity_2¢$lstm_cell_40/StatefulPartitionedCallÝ
$lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCallinputsstatesstates_1lstm_cell_40_1330208lstm_cell_40_1330210lstm_cell_40_1330212*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1330207}
IdentityIdentity-lstm_cell_40/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Identity_1Identity-lstm_cell_40/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Identity_2Identity-lstm_cell_40/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
NoOpNoOp%^lstm_cell_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_40/StatefulPartitionedCall$lstm_cell_40/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¾
È
while_cond_1330065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1330065___redundant_placeholder05
1while_while_cond_1330065___redundant_placeholder15
1while_while_cond_1330065___redundant_placeholder25
1while_while_cond_1330065___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ê:
É
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330820

inputs/
stacked_rnn_cells_12_1330737:	0
stacked_rnn_cells_12_1330739:
+
stacked_rnn_cells_12_1330741:	
identity

identity_1

identity_2¢,stacked_rnn_cells_12/StatefulPartitionedCall¢while;
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask§
,stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_12_1330737stacked_rnn_cells_12_1330739stacked_rnn_cells_12_1330741*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330220n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_12_1330737stacked_rnn_cells_12_1330739stacked_rnn_cells_12_1330741*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1330750*
condR
while_cond_1330749*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
NoOpNoOp-^stacked_rnn_cells_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ°	: : : 2\
,stacked_rnn_cells_12/StatefulPartitionedCall,stacked_rnn_cells_12/StatefulPartitionedCall2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
Ù$
¤
while_body_1330447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
$while_stacked_rnn_cells_12_1330471_0:	8
$while_stacked_rnn_cells_12_1330473_0:
3
$while_stacked_rnn_cells_12_1330475_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
"while_stacked_rnn_cells_12_1330471:	6
"while_stacked_rnn_cells_12_1330473:
1
"while_stacked_rnn_cells_12_1330475:	¢2while/stacked_rnn_cells_12/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0å
2while/stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3$while_stacked_rnn_cells_12_1330471_0$while_stacked_rnn_cells_12_1330473_0$while_stacked_rnn_cells_12_1330475_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330052ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp3^while/stacked_rnn_cells_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"J
"while_stacked_rnn_cells_12_1330471$while_stacked_rnn_cells_12_1330471_0"J
"while_stacked_rnn_cells_12_1330473$while_stacked_rnn_cells_12_1330473_0"J
"while_stacked_rnn_cells_12_1330475$while_stacked_rnn_cells_12_1330475_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2h
2while/stacked_rnn_cells_12/StatefulPartitionedCall2while/stacked_rnn_cells_12/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÕX
Ð
$enc_dec_12_rnn_12_while_body_1329847@
<enc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_loop_counterF
Benc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_maximum_iterations'
#enc_dec_12_rnn_12_while_placeholder)
%enc_dec_12_rnn_12_while_placeholder_1)
%enc_dec_12_rnn_12_while_placeholder_2)
%enc_dec_12_rnn_12_while_placeholder_3?
;enc_dec_12_rnn_12_while_enc_dec_12_rnn_12_strided_slice_1_0{
wenc_dec_12_rnn_12_while_tensorarrayv2read_tensorlistgetitem_enc_dec_12_rnn_12_tensorarrayunstack_tensorlistfromtensor_0m
Zenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0:	p
\enc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0:
j
[enc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0:	$
 enc_dec_12_rnn_12_while_identity&
"enc_dec_12_rnn_12_while_identity_1&
"enc_dec_12_rnn_12_while_identity_2&
"enc_dec_12_rnn_12_while_identity_3&
"enc_dec_12_rnn_12_while_identity_4&
"enc_dec_12_rnn_12_while_identity_5=
9enc_dec_12_rnn_12_while_enc_dec_12_rnn_12_strided_slice_1y
uenc_dec_12_rnn_12_while_tensorarrayv2read_tensorlistgetitem_enc_dec_12_rnn_12_tensorarrayunstack_tensorlistfromtensork
Xenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	n
Zenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
h
Yenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	¢Penc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢Oenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢Qenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp
Ienc_dec_12/rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;enc_dec_12/rnn_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemwenc_dec_12_rnn_12_while_tensorarrayv2read_tensorlistgetitem_enc_dec_12_rnn_12_tensorarrayunstack_tensorlistfromtensor_0#enc_dec_12_rnn_12_while_placeholderRenc_dec_12/rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ë
Oenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpZenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0
@enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMulBenc_dec_12/rnn_12/while/TensorArrayV2Read/TensorListGetItem:item:0Wenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
Qenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp\enc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
Benc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMul%enc_dec_12_rnn_12_while_placeholder_2Yenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/addAddV2Jenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0Lenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
Penc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp[enc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
Aenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAddAenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add:z:0Xenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ienc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :è
?enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/splitSplitRenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:0Jenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitÉ
Aenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoidHenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
Cenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1SigmoidHenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
=enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mulMulGenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0%enc_dec_12_rnn_12_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
>enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/TanhTanhHenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_1MulEenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:0Benc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1AddV2Aenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul:z:0Cenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
Cenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2SigmoidHenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
@enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1TanhCenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2MulGenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:0Denc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
<enc_dec_12/rnn_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem%enc_dec_12_rnn_12_while_placeholder_1#enc_dec_12_rnn_12_while_placeholderCenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒ_
enc_dec_12/rnn_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
enc_dec_12/rnn_12/while/addAddV2#enc_dec_12_rnn_12_while_placeholder&enc_dec_12/rnn_12/while/add/y:output:0*
T0*
_output_shapes
: a
enc_dec_12/rnn_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¯
enc_dec_12/rnn_12/while/add_1AddV2<enc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_loop_counter(enc_dec_12/rnn_12/while/add_1/y:output:0*
T0*
_output_shapes
: 
 enc_dec_12/rnn_12/while/IdentityIdentity!enc_dec_12/rnn_12/while/add_1:z:0^enc_dec_12/rnn_12/while/NoOp*
T0*
_output_shapes
: ²
"enc_dec_12/rnn_12/while/Identity_1IdentityBenc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_maximum_iterations^enc_dec_12/rnn_12/while/NoOp*
T0*
_output_shapes
: 
"enc_dec_12/rnn_12/while/Identity_2Identityenc_dec_12/rnn_12/while/add:z:0^enc_dec_12/rnn_12/while/NoOp*
T0*
_output_shapes
: Ï
"enc_dec_12/rnn_12/while/Identity_3IdentityLenc_dec_12/rnn_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^enc_dec_12/rnn_12/while/NoOp*
T0*
_output_shapes
: :éèÒÅ
"enc_dec_12/rnn_12/while/Identity_4IdentityCenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0^enc_dec_12/rnn_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"enc_dec_12/rnn_12/while/Identity_5IdentityCenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0^enc_dec_12/rnn_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
enc_dec_12/rnn_12/while/NoOpNoOpQ^enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpP^enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpR^enc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "x
9enc_dec_12_rnn_12_while_enc_dec_12_rnn_12_strided_slice_1;enc_dec_12_rnn_12_while_enc_dec_12_rnn_12_strided_slice_1_0"M
 enc_dec_12_rnn_12_while_identity)enc_dec_12/rnn_12/while/Identity:output:0"Q
"enc_dec_12_rnn_12_while_identity_1+enc_dec_12/rnn_12/while/Identity_1:output:0"Q
"enc_dec_12_rnn_12_while_identity_2+enc_dec_12/rnn_12/while/Identity_2:output:0"Q
"enc_dec_12_rnn_12_while_identity_3+enc_dec_12/rnn_12/while/Identity_3:output:0"Q
"enc_dec_12_rnn_12_while_identity_4+enc_dec_12/rnn_12/while/Identity_4:output:0"Q
"enc_dec_12_rnn_12_while_identity_5+enc_dec_12/rnn_12/while/Identity_5:output:0"¸
Yenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource[enc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0"º
Zenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource\enc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0"¶
Xenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceZenc_dec_12_rnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0"ð
uenc_dec_12_rnn_12_while_tensorarrayv2read_tensorlistgetitem_enc_dec_12_rnn_12_tensorarrayunstack_tensorlistfromtensorwenc_dec_12_rnn_12_while_tensorarrayv2read_tensorlistgetitem_enc_dec_12_rnn_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2¤
Penc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpPenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2¢
Oenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpOenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2¦
Qenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpQenc_dec_12/rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

Ø
(__inference_rnn_12_layer_call_fn_1331540
inputs_0
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330136p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÂC


while_body_1331932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0:	^
Jwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0:
X
Iwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	\
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
V
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	¢>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ç
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ä
.while/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ë
0while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+while/stacked_rnn_cells_12/lstm_cell_40/addAddV28while/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0:while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0æ
/while/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd/while/stacked_rnn_cells_12/lstm_cell_40/add:z:0Fwhile/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7while/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :²
-while/stacked_rnn_cells_12/lstm_cell_40/splitSplit@while/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:08while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¥
/while/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
+while/stacked_rnn_cells_12/lstm_cell_40/mulMul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,while/stacked_rnn_cells_12/lstm_cell_40/TanhTanh6while/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul3while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:00while/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
-while/stacked_rnn_cells_12/lstm_cell_40/add_1AddV2/while/stacked_rnn_cells_12/lstm_cell_40/mul:z:01while/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:02while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp?^while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp@^while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0"
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0"
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
·
°
$enc_dec_12_rnn_12_while_cond_1329846@
<enc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_loop_counterF
Benc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_maximum_iterations'
#enc_dec_12_rnn_12_while_placeholder)
%enc_dec_12_rnn_12_while_placeholder_1)
%enc_dec_12_rnn_12_while_placeholder_2)
%enc_dec_12_rnn_12_while_placeholder_3B
>enc_dec_12_rnn_12_while_less_enc_dec_12_rnn_12_strided_slice_1Y
Uenc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_cond_1329846___redundant_placeholder0Y
Uenc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_cond_1329846___redundant_placeholder1Y
Uenc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_cond_1329846___redundant_placeholder2Y
Uenc_dec_12_rnn_12_while_enc_dec_12_rnn_12_while_cond_1329846___redundant_placeholder3$
 enc_dec_12_rnn_12_while_identity
ª
enc_dec_12/rnn_12/while/LessLess#enc_dec_12_rnn_12_while_placeholder>enc_dec_12_rnn_12_while_less_enc_dec_12_rnn_12_strided_slice_1*
T0*
_output_shapes
: o
 enc_dec_12/rnn_12/while/IdentityIdentity enc_dec_12/rnn_12/while/Less:z:0*
T0
*
_output_shapes
: "M
 enc_dec_12_rnn_12_while_identity)enc_dec_12/rnn_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
à

I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1332246

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ø

I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1330207

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
û
ø
.__inference_lstm_cell_40_layer_call_fn_1332214

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1330207p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
é	
É
,__inference_enc_dec_12_layer_call_fn_1330930
input_1
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330890s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
!
_user_specified_name	input_1
û
â
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1331525

inputs

states_0_0

states_0_1>
+lstm_cell_40_matmul_readvariableop_resource:	A
-lstm_cell_40_matmul_1_readvariableop_resource:
;
,lstm_cell_40_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢#lstm_cell_40/BiasAdd/ReadVariableOp¢"lstm_cell_40/MatMul/ReadVariableOp¢$lstm_cell_40/MatMul_1/ReadVariableOp
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_40/MatMulMatMulinputs*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_40/MatMul_1MatMul
states_0_0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0
states_0_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_40/TanhTanhlstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_40/Tanh_1Tanhlstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentitylstm_cell_40/mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identitylstm_cell_40/mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_2Identitylstm_cell_40/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1
æ	
È
,__inference_enc_dec_12_layer_call_fn_1331025

inputs
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330593s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs


Ô
rnn_12_while_cond_1331283*
&rnn_12_while_rnn_12_while_loop_counter0
,rnn_12_while_rnn_12_while_maximum_iterations
rnn_12_while_placeholder
rnn_12_while_placeholder_1
rnn_12_while_placeholder_2
rnn_12_while_placeholder_3,
(rnn_12_while_less_rnn_12_strided_slice_1C
?rnn_12_while_rnn_12_while_cond_1331283___redundant_placeholder0C
?rnn_12_while_rnn_12_while_cond_1331283___redundant_placeholder1C
?rnn_12_while_rnn_12_while_cond_1331283___redundant_placeholder2C
?rnn_12_while_rnn_12_while_cond_1331283___redundant_placeholder3
rnn_12_while_identity
~
rnn_12/while/LessLessrnn_12_while_placeholder(rnn_12_while_less_rnn_12_strided_slice_1*
T0*
_output_shapes
: Y
rnn_12/while/IdentityIdentityrnn_12/while/Less:z:0*
T0
*
_output_shapes
: "7
rnn_12_while_identityrnn_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


6__inference_stacked_rnn_cells_12_layer_call_fn_1331444

inputs

states_0_0

states_0_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall»
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330052p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1
¾
È
while_cond_1332075
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1332075___redundant_placeholder05
1while_while_cond_1332075___redundant_placeholder15
1while_while_cond_1332075___redundant_placeholder25
1while_while_cond_1332075___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ê

*__inference_dense_14_layer_call_fn_1332170

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1330583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
Ï	
"__inference__wrapped_model_1329967
input_1e
Renc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	h
Tenc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
b
Senc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	I
6enc_dec_12_lstm_cell_41_matmul_readvariableop_resource:	L
8enc_dec_12_lstm_cell_41_matmul_1_readvariableop_resource:
F
7enc_dec_12_lstm_cell_41_biasadd_readvariableop_resource:	E
2enc_dec_12_dense_14_matmul_readvariableop_resource:	A
3enc_dec_12_dense_14_biasadd_readvariableop_resource:
identity¢*enc_dec_12/dense_14/BiasAdd/ReadVariableOp¢)enc_dec_12/dense_14/MatMul/ReadVariableOp¢.enc_dec_12/lstm_cell_41/BiasAdd/ReadVariableOp¢-enc_dec_12/lstm_cell_41/MatMul/ReadVariableOp¢/enc_dec_12/lstm_cell_41/MatMul_1/ReadVariableOp¢Jenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢Ienc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢Kenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp¢enc_dec_12/rnn_12/whileN
enc_dec_12/rnn_12/ShapeShapeinput_1*
T0*
_output_shapes
:o
%enc_dec_12/rnn_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'enc_dec_12/rnn_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'enc_dec_12/rnn_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
enc_dec_12/rnn_12/strided_sliceStridedSlice enc_dec_12/rnn_12/Shape:output:0.enc_dec_12/rnn_12/strided_slice/stack:output:00enc_dec_12/rnn_12/strided_slice/stack_1:output:00enc_dec_12/rnn_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 enc_dec_12/rnn_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :©
enc_dec_12/rnn_12/zeros/packedPack(enc_dec_12/rnn_12/strided_slice:output:0)enc_dec_12/rnn_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:b
enc_dec_12/rnn_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    £
enc_dec_12/rnn_12/zerosFill'enc_dec_12/rnn_12/zeros/packed:output:0&enc_dec_12/rnn_12/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
"enc_dec_12/rnn_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :­
 enc_dec_12/rnn_12/zeros_1/packedPack(enc_dec_12/rnn_12/strided_slice:output:0+enc_dec_12/rnn_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:d
enc_dec_12/rnn_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ©
enc_dec_12/rnn_12/zeros_1Fill)enc_dec_12/rnn_12/zeros_1/packed:output:0(enc_dec_12/rnn_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 enc_dec_12/rnn_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
enc_dec_12/rnn_12/transpose	Transposeinput_1)enc_dec_12/rnn_12/transpose/perm:output:0*
T0*,
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿh
enc_dec_12/rnn_12/Shape_1Shapeenc_dec_12/rnn_12/transpose:y:0*
T0*
_output_shapes
:q
'enc_dec_12/rnn_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)enc_dec_12/rnn_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)enc_dec_12/rnn_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!enc_dec_12/rnn_12/strided_slice_1StridedSlice"enc_dec_12/rnn_12/Shape_1:output:00enc_dec_12/rnn_12/strided_slice_1/stack:output:02enc_dec_12/rnn_12/strided_slice_1/stack_1:output:02enc_dec_12/rnn_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-enc_dec_12/rnn_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
enc_dec_12/rnn_12/TensorArrayV2TensorListReserve6enc_dec_12/rnn_12/TensorArrayV2/element_shape:output:0*enc_dec_12/rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Genc_dec_12/rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
9enc_dec_12/rnn_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorenc_dec_12/rnn_12/transpose:y:0Penc_dec_12/rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒq
'enc_dec_12/rnn_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)enc_dec_12/rnn_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)enc_dec_12/rnn_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
!enc_dec_12/rnn_12/strided_slice_2StridedSliceenc_dec_12/rnn_12/transpose:y:00enc_dec_12/rnn_12/strided_slice_2/stack:output:02enc_dec_12/rnn_12/strided_slice_2/stack_1:output:02enc_dec_12/rnn_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÝ
Ienc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpRenc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0ö
:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMul*enc_dec_12/rnn_12/strided_slice_2:output:0Qenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
Kenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpTenc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0ð
<enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMul enc_dec_12/rnn_12/zeros:output:0Senc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/addAddV2Denc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0Fenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
Jenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpSenc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
;enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd;enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/add:z:0Renc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Cenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ö
9enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/splitSplitLenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:0Denc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split½
;enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoidBenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
=enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1SigmoidBenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
7enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/mulMulAenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0"enc_dec_12/rnn_12/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/TanhTanhBenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
9enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul?enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:0<enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
9enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/add_1AddV2;enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul:z:0=enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
=enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2SigmoidBenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh=enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
9enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_2MulAenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:0>enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/enc_dec_12/rnn_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   î
!enc_dec_12/rnn_12/TensorArrayV2_1TensorListReserve8enc_dec_12/rnn_12/TensorArrayV2_1/element_shape:output:0*enc_dec_12/rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
enc_dec_12/rnn_12/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*enc_dec_12/rnn_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿf
$enc_dec_12/rnn_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ã
enc_dec_12/rnn_12/whileWhile-enc_dec_12/rnn_12/while/loop_counter:output:03enc_dec_12/rnn_12/while/maximum_iterations:output:0enc_dec_12/rnn_12/time:output:0*enc_dec_12/rnn_12/TensorArrayV2_1:handle:0 enc_dec_12/rnn_12/zeros:output:0"enc_dec_12/rnn_12/zeros_1:output:0*enc_dec_12/rnn_12/strided_slice_1:output:0Ienc_dec_12/rnn_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0Renc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceTenc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceSenc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$enc_dec_12_rnn_12_while_body_1329847*0
cond(R&
$enc_dec_12_rnn_12_while_cond_1329846*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Benc_dec_12/rnn_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ú
4enc_dec_12/rnn_12/TensorArrayV2Stack/TensorListStackTensorListStack enc_dec_12/rnn_12/while:output:3Kenc_dec_12/rnn_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿ*
element_dtype0z
'enc_dec_12/rnn_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿs
)enc_dec_12/rnn_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)enc_dec_12/rnn_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
!enc_dec_12/rnn_12/strided_slice_3StridedSlice=enc_dec_12/rnn_12/TensorArrayV2Stack/TensorListStack:tensor:00enc_dec_12/rnn_12/strided_slice_3/stack:output:02enc_dec_12/rnn_12/strided_slice_3/stack_1:output:02enc_dec_12/rnn_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskw
"enc_dec_12/rnn_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Î
enc_dec_12/rnn_12/transpose_1	Transpose=enc_dec_12/rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0+enc_dec_12/rnn_12/transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	s
enc_dec_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    u
 enc_dec_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           u
 enc_dec_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ©
enc_dec_12/strided_sliceStridedSliceinput_1'enc_dec_12/strided_slice/stack:output:0)enc_dec_12/strided_slice/stack_1:output:0)enc_dec_12/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask¥
-enc_dec_12/lstm_cell_41/MatMul/ReadVariableOpReadVariableOp6enc_dec_12_lstm_cell_41_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0µ
enc_dec_12/lstm_cell_41/MatMulMatMul!enc_dec_12/strided_slice:output:05enc_dec_12/lstm_cell_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
/enc_dec_12/lstm_cell_41/MatMul_1/ReadVariableOpReadVariableOp8enc_dec_12_lstm_cell_41_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0¸
 enc_dec_12/lstm_cell_41/MatMul_1MatMul enc_dec_12/rnn_12/while:output:47enc_dec_12/lstm_cell_41/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
enc_dec_12/lstm_cell_41/addAddV2(enc_dec_12/lstm_cell_41/MatMul:product:0*enc_dec_12/lstm_cell_41/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.enc_dec_12/lstm_cell_41/BiasAdd/ReadVariableOpReadVariableOp7enc_dec_12_lstm_cell_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
enc_dec_12/lstm_cell_41/BiasAddBiasAddenc_dec_12/lstm_cell_41/add:z:06enc_dec_12/lstm_cell_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'enc_dec_12/lstm_cell_41/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
enc_dec_12/lstm_cell_41/splitSplit0enc_dec_12/lstm_cell_41/split/split_dim:output:0(enc_dec_12/lstm_cell_41/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
enc_dec_12/lstm_cell_41/SigmoidSigmoid&enc_dec_12/lstm_cell_41/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!enc_dec_12/lstm_cell_41/Sigmoid_1Sigmoid&enc_dec_12/lstm_cell_41/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_dec_12/lstm_cell_41/mulMul%enc_dec_12/lstm_cell_41/Sigmoid_1:y:0 enc_dec_12/rnn_12/while:output:5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_dec_12/lstm_cell_41/TanhTanh&enc_dec_12/lstm_cell_41/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_dec_12/lstm_cell_41/mul_1Mul#enc_dec_12/lstm_cell_41/Sigmoid:y:0 enc_dec_12/lstm_cell_41/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_dec_12/lstm_cell_41/add_1AddV2enc_dec_12/lstm_cell_41/mul:z:0!enc_dec_12/lstm_cell_41/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!enc_dec_12/lstm_cell_41/Sigmoid_2Sigmoid&enc_dec_12/lstm_cell_41/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
enc_dec_12/lstm_cell_41/Tanh_1Tanh!enc_dec_12/lstm_cell_41/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
enc_dec_12/lstm_cell_41/mul_2Mul%enc_dec_12/lstm_cell_41/Sigmoid_2:y:0"enc_dec_12/lstm_cell_41/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)enc_dec_12/dense_14/MatMul/ReadVariableOpReadVariableOp2enc_dec_12_dense_14_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¬
enc_dec_12/dense_14/MatMulMatMul!enc_dec_12/lstm_cell_41/mul_2:z:01enc_dec_12/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*enc_dec_12/dense_14/BiasAdd/ReadVariableOpReadVariableOp3enc_dec_12_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
enc_dec_12/dense_14/BiasAddBiasAdd$enc_dec_12/dense_14/MatMul:product:02enc_dec_12/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
enc_dec_12/stackPack$enc_dec_12/dense_14/BiasAdd:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
enc_dec_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
enc_dec_12/transpose	Transposeenc_dec_12/stack:output:0"enc_dec_12/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityenc_dec_12/transpose:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
NoOpNoOp+^enc_dec_12/dense_14/BiasAdd/ReadVariableOp*^enc_dec_12/dense_14/MatMul/ReadVariableOp/^enc_dec_12/lstm_cell_41/BiasAdd/ReadVariableOp.^enc_dec_12/lstm_cell_41/MatMul/ReadVariableOp0^enc_dec_12/lstm_cell_41/MatMul_1/ReadVariableOpK^enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpJ^enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpL^enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp^enc_dec_12/rnn_12/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 2X
*enc_dec_12/dense_14/BiasAdd/ReadVariableOp*enc_dec_12/dense_14/BiasAdd/ReadVariableOp2V
)enc_dec_12/dense_14/MatMul/ReadVariableOp)enc_dec_12/dense_14/MatMul/ReadVariableOp2`
.enc_dec_12/lstm_cell_41/BiasAdd/ReadVariableOp.enc_dec_12/lstm_cell_41/BiasAdd/ReadVariableOp2^
-enc_dec_12/lstm_cell_41/MatMul/ReadVariableOp-enc_dec_12/lstm_cell_41/MatMul/ReadVariableOp2b
/enc_dec_12/lstm_cell_41/MatMul_1/ReadVariableOp/enc_dec_12/lstm_cell_41/MatMul_1/ReadVariableOp2
Jenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpJenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2
Ienc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpIenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
Kenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpKenc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp22
enc_dec_12/rnn_12/whileenc_dec_12/rnn_12/while:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
!
_user_specified_name	input_1
¾
È
while_cond_1330446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1330446___redundant_placeholder05
1while_while_cond_1330446___redundant_placeholder15
1while_while_cond_1330446___redundant_placeholder25
1while_while_cond_1330446___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÂC


while_body_1331788
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0:	^
Jwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0:
X
Iwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	\
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
V
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	¢>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ç
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ä
.while/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ë
0while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+while/stacked_rnn_cells_12/lstm_cell_40/addAddV28while/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0:while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0æ
/while/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd/while/stacked_rnn_cells_12/lstm_cell_40/add:z:0Fwhile/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7while/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :²
-while/stacked_rnn_cells_12/lstm_cell_40/splitSplit@while/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:08while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¥
/while/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
+while/stacked_rnn_cells_12/lstm_cell_40/mulMul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,while/stacked_rnn_cells_12/lstm_cell_40/TanhTanh6while/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul3while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:00while/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
-while/stacked_rnn_cells_12/lstm_cell_40/add_1AddV2/while/stacked_rnn_cells_12/lstm_cell_40/mul:z:01while/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:02while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp?^while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp@^while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0"
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0"
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ø

I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330563

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ì	
÷
E__inference_dense_14_layer_call_and_return_conditional_losses_1330583

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

Ö
(__inference_rnn_12_layer_call_fn_1331570

inputs
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330517p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ°	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
à

I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1332376

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
º
¸
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1331404

inputsZ
Grnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	]
Irnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
W
Hrnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	>
+lstm_cell_41_matmul_readvariableop_resource:	A
-lstm_cell_41_matmul_1_readvariableop_resource:
;
,lstm_cell_41_biasadd_readvariableop_resource:	:
'dense_14_matmul_readvariableop_resource:	6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢#lstm_cell_41/BiasAdd/ReadVariableOp¢"lstm_cell_41/MatMul/ReadVariableOp¢$lstm_cell_41/MatMul_1/ReadVariableOp¢?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp¢rnn_12/whileB
rnn_12/ShapeShapeinputs*
T0*
_output_shapes
:d
rnn_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
rnn_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
rnn_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
rnn_12/strided_sliceStridedSlicernn_12/Shape:output:0#rnn_12/strided_slice/stack:output:0%rnn_12/strided_slice/stack_1:output:0%rnn_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
rnn_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn_12/zeros/packedPackrnn_12/strided_slice:output:0rnn_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
rnn_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_12/zerosFillrnn_12/zeros/packed:output:0rnn_12/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rnn_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn_12/zeros_1/packedPackrnn_12/strided_slice:output:0 rnn_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
rnn_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_12/zeros_1Fillrnn_12/zeros_1/packed:output:0rnn_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
rnn_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rnn_12/transpose	Transposeinputsrnn_12/transpose/perm:output:0*
T0*,
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿR
rnn_12/Shape_1Shapernn_12/transpose:y:0*
T0*
_output_shapes
:f
rnn_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
rnn_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
rnn_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
rnn_12/strided_slice_1StridedSlicernn_12/Shape_1:output:0%rnn_12/strided_slice_1/stack:output:0'rnn_12/strided_slice_1/stack_1:output:0'rnn_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"rnn_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
rnn_12/TensorArrayV2TensorListReserve+rnn_12/TensorArrayV2/element_shape:output:0rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.rnn_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_12/transpose:y:0Ernn_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
rnn_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
rnn_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
rnn_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn_12/strided_slice_2StridedSlicernn_12/transpose:y:0%rnn_12/strided_slice_2/stack:output:0'rnn_12/strided_slice_2/stack_1:output:0'rnn_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÇ
>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpGrnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Õ
/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMulrnn_12/strided_slice_2:output:0Frnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpIrnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Ï
1rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulrnn_12/zeros:output:0Hrnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
,rnn_12/stacked_rnn_cells_12/lstm_cell_40/addAddV29rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0;rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpHrnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0é
0rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd0rnn_12/stacked_rnn_cells_12/lstm_cell_40/add:z:0Grnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8rnn_12/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :µ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/splitSplitArnn_12/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:09rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split§
0rnn_12/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
2rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
,rnn_12/stacked_rnn_cells_12/lstm_cell_40/mulMul6rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0rnn_12/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-rnn_12/stacked_rnn_cells_12/lstm_cell_40/TanhTanh7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul4rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:01rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/add_1AddV20rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul:z:02rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
2rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh2rnn_12/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul6rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:03rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$rnn_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
rnn_12/TensorArrayV2_1TensorListReserve-rnn_12/TensorArrayV2_1/element_shape:output:0rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
rnn_12/timeConst*
_output_shapes
: *
dtype0*
value	B : j
rnn_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
rnn_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
rnn_12/whileWhile"rnn_12/while/loop_counter:output:0(rnn_12/while/maximum_iterations:output:0rnn_12/time:output:0rnn_12/TensorArrayV2_1:handle:0rnn_12/zeros:output:0rnn_12/zeros_1:output:0rnn_12/strided_slice_1:output:0>rnn_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0Grnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceIrnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceHrnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
rnn_12_while_body_1331284*%
condR
rnn_12_while_cond_1331283*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7rnn_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ù
)rnn_12/TensorArrayV2Stack/TensorListStackTensorListStackrnn_12/while:output:3@rnn_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
rnn_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
rnn_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
rnn_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
rnn_12/strided_slice_3StridedSlice2rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0%rnn_12/strided_slice_3/stack:output:0'rnn_12/strided_slice_3/stack_1:output:0'rnn_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
rnn_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
rnn_12/transpose_1	Transpose2rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0 rnn_12/transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ü
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
"lstm_cell_41/MatMul/ReadVariableOpReadVariableOp+lstm_cell_41_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_41/MatMulMatMulstrided_slice:output:0*lstm_cell_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_41/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_41_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_41/MatMul_1MatMulrnn_12/while:output:4,lstm_cell_41/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_41/addAddV2lstm_cell_41/MatMul:product:0lstm_cell_41/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_41/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_41/BiasAddBiasAddlstm_cell_41/add:z:0+lstm_cell_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_41/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_41/splitSplit%lstm_cell_41/split/split_dim:output:0lstm_cell_41/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_41/SigmoidSigmoidlstm_cell_41/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_41/Sigmoid_1Sigmoidlstm_cell_41/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_41/mulMullstm_cell_41/Sigmoid_1:y:0rnn_12/while:output:5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_41/TanhTanhlstm_cell_41/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_41/mul_1Mullstm_cell_41/Sigmoid:y:0lstm_cell_41/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_41/add_1AddV2lstm_cell_41/mul:z:0lstm_cell_41/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_41/Sigmoid_2Sigmoidlstm_cell_41/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_41/Tanh_1Tanhlstm_cell_41/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_41/mul_2Mullstm_cell_41/Sigmoid_2:y:0lstm_cell_41/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_14/MatMulMatMullstm_cell_41/mul_2:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
stackPackdense_14/BiasAdd:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp$^lstm_cell_41/BiasAdd/ReadVariableOp#^lstm_cell_41/MatMul/ReadVariableOp%^lstm_cell_41/MatMul_1/ReadVariableOp@^rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp?^rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpA^rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp^rnn_12/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2J
#lstm_cell_41/BiasAdd/ReadVariableOp#lstm_cell_41/BiasAdd/ReadVariableOp2H
"lstm_cell_41/MatMul/ReadVariableOp"lstm_cell_41/MatMul/ReadVariableOp2L
$lstm_cell_41/MatMul_1/ReadVariableOp$lstm_cell_41/MatMul_1/ReadVariableOp2
?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2
>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp2
rnn_12/whilernn_12/while:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
÷

Ö
(__inference_rnn_12_layer_call_fn_1331585

inputs
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330820p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ°	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
×
õ
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330998
input_1!
rnn_12_1330967:	"
rnn_12_1330969:

rnn_12_1330971:	'
lstm_cell_41_1330980:	(
lstm_cell_41_1330982:
#
lstm_cell_41_1330984:	#
dense_14_1330989:	
dense_14_1330991:
identity¢ dense_14/StatefulPartitionedCall¢$lstm_cell_41/StatefulPartitionedCall¢rnn_12/StatefulPartitionedCall¬
rnn_12/StatefulPartitionedCallStatefulPartitionedCallinput_1rnn_12_1330967rnn_12_1330969rnn_12_1330971*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330820h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ý
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask­
$lstm_cell_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0'rnn_12/StatefulPartitionedCall:output:1'rnn_12/StatefulPartitionedCall:output:2lstm_cell_41_1330980lstm_cell_41_1330982lstm_cell_41_1330984*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330676
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-lstm_cell_41/StatefulPartitionedCall:output:0dense_14_1330989dense_14_1330991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1330583w
stackPack)dense_14/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall%^lstm_cell_41/StatefulPartitionedCall^rnn_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$lstm_cell_41/StatefulPartitionedCall$lstm_cell_41/StatefulPartitionedCall2@
rnn_12/StatefulPartitionedCallrnn_12/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
!
_user_specified_name	input_1
ò:
É
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330136

inputs/
stacked_rnn_cells_12_1330053:	0
stacked_rnn_cells_12_1330055:
+
stacked_rnn_cells_12_1330057:	
identity

identity_1

identity_2¢,stacked_rnn_cells_12/StatefulPartitionedCall¢while;
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask§
,stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_12_1330053stacked_rnn_cells_12_1330055stacked_rnn_cells_12_1330057*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330052n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_12_1330053stacked_rnn_cells_12_1330055stacked_rnn_cells_12_1330057*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1330066*
condR
while_cond_1330065*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
NoOpNoOp-^stacked_rnn_cells_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2\
,stacked_rnn_cells_12/StatefulPartitionedCall,stacked_rnn_cells_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
õ
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330964
input_1!
rnn_12_1330933:	"
rnn_12_1330935:

rnn_12_1330937:	'
lstm_cell_41_1330946:	(
lstm_cell_41_1330948:
#
lstm_cell_41_1330950:	#
dense_14_1330955:	
dense_14_1330957:
identity¢ dense_14/StatefulPartitionedCall¢$lstm_cell_41/StatefulPartitionedCall¢rnn_12/StatefulPartitionedCall¬
rnn_12/StatefulPartitionedCallStatefulPartitionedCallinput_1rnn_12_1330933rnn_12_1330935rnn_12_1330937*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330517h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ý
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask­
$lstm_cell_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0'rnn_12/StatefulPartitionedCall:output:1'rnn_12/StatefulPartitionedCall:output:2lstm_cell_41_1330946lstm_cell_41_1330948lstm_cell_41_1330950*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330563
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-lstm_cell_41/StatefulPartitionedCall:output:0dense_14_1330955dense_14_1330957*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1330583w
stackPack)dense_14/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall%^lstm_cell_41/StatefulPartitionedCall^rnn_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$lstm_cell_41/StatefulPartitionedCall$lstm_cell_41/StatefulPartitionedCall2@
rnn_12/StatefulPartitionedCallrnn_12/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
!
_user_specified_name	input_1
º
¸
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1331225

inputsZ
Grnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	]
Irnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
W
Hrnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	>
+lstm_cell_41_matmul_readvariableop_resource:	A
-lstm_cell_41_matmul_1_readvariableop_resource:
;
,lstm_cell_41_biasadd_readvariableop_resource:	:
'dense_14_matmul_readvariableop_resource:	6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢#lstm_cell_41/BiasAdd/ReadVariableOp¢"lstm_cell_41/MatMul/ReadVariableOp¢$lstm_cell_41/MatMul_1/ReadVariableOp¢?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp¢rnn_12/whileB
rnn_12/ShapeShapeinputs*
T0*
_output_shapes
:d
rnn_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
rnn_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
rnn_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
rnn_12/strided_sliceStridedSlicernn_12/Shape:output:0#rnn_12/strided_slice/stack:output:0%rnn_12/strided_slice/stack_1:output:0%rnn_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
rnn_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn_12/zeros/packedPackrnn_12/strided_slice:output:0rnn_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
rnn_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_12/zerosFillrnn_12/zeros/packed:output:0rnn_12/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rnn_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn_12/zeros_1/packedPackrnn_12/strided_slice:output:0 rnn_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
rnn_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn_12/zeros_1Fillrnn_12/zeros_1/packed:output:0rnn_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
rnn_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rnn_12/transpose	Transposeinputsrnn_12/transpose/perm:output:0*
T0*,
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿR
rnn_12/Shape_1Shapernn_12/transpose:y:0*
T0*
_output_shapes
:f
rnn_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
rnn_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
rnn_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
rnn_12/strided_slice_1StridedSlicernn_12/Shape_1:output:0%rnn_12/strided_slice_1/stack:output:0'rnn_12/strided_slice_1/stack_1:output:0'rnn_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"rnn_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
rnn_12/TensorArrayV2TensorListReserve+rnn_12/TensorArrayV2/element_shape:output:0rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.rnn_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_12/transpose:y:0Ernn_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
rnn_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
rnn_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
rnn_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn_12/strided_slice_2StridedSlicernn_12/transpose:y:0%rnn_12/strided_slice_2/stack:output:0'rnn_12/strided_slice_2/stack_1:output:0'rnn_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÇ
>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpGrnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Õ
/rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMulrnn_12/strided_slice_2:output:0Frnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpIrnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Ï
1rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulrnn_12/zeros:output:0Hrnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
,rnn_12/stacked_rnn_cells_12/lstm_cell_40/addAddV29rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0;rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpHrnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0é
0rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd0rnn_12/stacked_rnn_cells_12/lstm_cell_40/add:z:0Grnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8rnn_12/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :µ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/splitSplitArnn_12/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:09rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split§
0rnn_12/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
2rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
,rnn_12/stacked_rnn_cells_12/lstm_cell_40/mulMul6rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0rnn_12/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-rnn_12/stacked_rnn_cells_12/lstm_cell_40/TanhTanh7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul4rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:01rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/add_1AddV20rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul:z:02rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
2rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid7rnn_12/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh2rnn_12/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
.rnn_12/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul6rnn_12/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:03rnn_12/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$rnn_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Í
rnn_12/TensorArrayV2_1TensorListReserve-rnn_12/TensorArrayV2_1/element_shape:output:0rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
rnn_12/timeConst*
_output_shapes
: *
dtype0*
value	B : j
rnn_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
rnn_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ©
rnn_12/whileWhile"rnn_12/while/loop_counter:output:0(rnn_12/while/maximum_iterations:output:0rnn_12/time:output:0rnn_12/TensorArrayV2_1:handle:0rnn_12/zeros:output:0rnn_12/zeros_1:output:0rnn_12/strided_slice_1:output:0>rnn_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0Grnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceIrnn_12_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceHrnn_12_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
rnn_12_while_body_1331105*%
condR
rnn_12_while_cond_1331104*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7rnn_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ù
)rnn_12/TensorArrayV2Stack/TensorListStackTensorListStackrnn_12/while:output:3@rnn_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿ*
element_dtype0o
rnn_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
rnn_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
rnn_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
rnn_12/strided_slice_3StridedSlice2rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0%rnn_12/strided_slice_3/stack:output:0'rnn_12/strided_slice_3/stack_1:output:0'rnn_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
rnn_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
rnn_12/transpose_1	Transpose2rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0 rnn_12/transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ü
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
"lstm_cell_41/MatMul/ReadVariableOpReadVariableOp+lstm_cell_41_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_41/MatMulMatMulstrided_slice:output:0*lstm_cell_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_41/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_41_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_41/MatMul_1MatMulrnn_12/while:output:4,lstm_cell_41/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_41/addAddV2lstm_cell_41/MatMul:product:0lstm_cell_41/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_41/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_41_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_41/BiasAddBiasAddlstm_cell_41/add:z:0+lstm_cell_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_41/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_41/splitSplit%lstm_cell_41/split/split_dim:output:0lstm_cell_41/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_41/SigmoidSigmoidlstm_cell_41/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_41/Sigmoid_1Sigmoidlstm_cell_41/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_41/mulMullstm_cell_41/Sigmoid_1:y:0rnn_12/while:output:5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_41/TanhTanhlstm_cell_41/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_41/mul_1Mullstm_cell_41/Sigmoid:y:0lstm_cell_41/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_41/add_1AddV2lstm_cell_41/mul:z:0lstm_cell_41/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_41/Sigmoid_2Sigmoidlstm_cell_41/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_41/Tanh_1Tanhlstm_cell_41/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_41/mul_2Mullstm_cell_41/Sigmoid_2:y:0lstm_cell_41/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_14/MatMulMatMullstm_cell_41/mul_2:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
stackPackdense_14/BiasAdd:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp$^lstm_cell_41/BiasAdd/ReadVariableOp#^lstm_cell_41/MatMul/ReadVariableOp%^lstm_cell_41/MatMul_1/ReadVariableOp@^rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp?^rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpA^rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp^rnn_12/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2J
#lstm_cell_41/BiasAdd/ReadVariableOp#lstm_cell_41/BiasAdd/ReadVariableOp2H
"lstm_cell_41/MatMul/ReadVariableOp"lstm_cell_41/MatMul/ReadVariableOp2L
$lstm_cell_41/MatMul_1/ReadVariableOp$lstm_cell_41/MatMul_1/ReadVariableOp2
?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp?rnn_12/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2
>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp>rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp@rnn_12/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp2
rnn_12/whilernn_12/while:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
¾
È
while_cond_1331787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1331787___redundant_placeholder05
1while_while_cond_1331787___redundant_placeholder15
1while_while_cond_1331787___redundant_placeholder25
1while_while_cond_1331787___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
à

I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1332344

inputs
states_0
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
½	
Â
%__inference_signature_wrapper_1331427
input_1
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1329967s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
!
_user_specified_name	input_1
í
ú
#__inference__traced_restore_1332619
file_prefix>
+assignvariableop_enc_dec_12_dense_14_kernel:	9
+assignvariableop_1_enc_dec_12_dense_14_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: `
Massignvariableop_7_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel:	k
Wassignvariableop_8_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel:
Z
Kassignvariableop_9_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias:	E
2assignvariableop_10_enc_dec_12_lstm_cell_41_kernel:	P
<assignvariableop_11_enc_dec_12_lstm_cell_41_recurrent_kernel:
?
0assignvariableop_12_enc_dec_12_lstm_cell_41_bias:	#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: %
assignvariableop_17_total_2: %
assignvariableop_18_count_2: H
5assignvariableop_19_adam_enc_dec_12_dense_14_kernel_m:	A
3assignvariableop_20_adam_enc_dec_12_dense_14_bias_m:h
Uassignvariableop_21_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_m:	s
_assignvariableop_22_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_m:
b
Sassignvariableop_23_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_m:	L
9assignvariableop_24_adam_enc_dec_12_lstm_cell_41_kernel_m:	W
Cassignvariableop_25_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_m:
F
7assignvariableop_26_adam_enc_dec_12_lstm_cell_41_bias_m:	H
5assignvariableop_27_adam_enc_dec_12_dense_14_kernel_v:	A
3assignvariableop_28_adam_enc_dec_12_dense_14_bias_v:h
Uassignvariableop_29_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_v:	s
_assignvariableop_30_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_v:
b
Sassignvariableop_31_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_v:	L
9assignvariableop_32_adam_enc_dec_12_lstm_cell_41_kernel_v:	W
Cassignvariableop_33_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_v:
F
7assignvariableop_34_adam_enc_dec_12_lstm_cell_41_bias_v:	
identity_36¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¶
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ü
valueÒBÏ$B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Õ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp+assignvariableop_enc_dec_12_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_enc_dec_12_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_7AssignVariableOpMassignvariableop_7_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_8AssignVariableOpWassignvariableop_8_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_9AssignVariableOpKassignvariableop_9_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_10AssignVariableOp2assignvariableop_10_enc_dec_12_lstm_cell_41_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_11AssignVariableOp<assignvariableop_11_enc_dec_12_lstm_cell_41_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_enc_dec_12_lstm_cell_41_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_19AssignVariableOp5assignvariableop_19_adam_enc_dec_12_dense_14_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_20AssignVariableOp3assignvariableop_20_adam_enc_dec_12_dense_14_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_21AssignVariableOpUassignvariableop_21_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_22AssignVariableOp_assignvariableop_22_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_23AssignVariableOpSassignvariableop_23_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_24AssignVariableOp9assignvariableop_24_adam_enc_dec_12_lstm_cell_41_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_25AssignVariableOpCassignvariableop_25_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_26AssignVariableOp7assignvariableop_26_adam_enc_dec_12_lstm_cell_41_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_enc_dec_12_dense_14_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_enc_dec_12_dense_14_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_29AssignVariableOpUassignvariableop_29_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_30AssignVariableOp_assignvariableop_30_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_31AssignVariableOpSassignvariableop_31_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_32AssignVariableOp9assignvariableop_32_adam_enc_dec_12_lstm_cell_41_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_33AssignVariableOpCassignvariableop_33_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_enc_dec_12_lstm_cell_41_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ñ
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: ¾
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
ÂC


while_body_1332076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0:	^
Jwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0:
X
Iwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	\
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
V
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	¢>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ç
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ä
.while/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ë
0while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+while/stacked_rnn_cells_12/lstm_cell_40/addAddV28while/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0:while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0æ
/while/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd/while/stacked_rnn_cells_12/lstm_cell_40/add:z:0Fwhile/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7while/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :²
-while/stacked_rnn_cells_12/lstm_cell_40/splitSplit@while/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:08while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¥
/while/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
+while/stacked_rnn_cells_12/lstm_cell_40/mulMul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,while/stacked_rnn_cells_12/lstm_cell_40/TanhTanh6while/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul3while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:00while/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
-while/stacked_rnn_cells_12/lstm_cell_40/add_1AddV2/while/stacked_rnn_cells_12/lstm_cell_40/mul:z:01while/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:02while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp?^while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp@^while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0"
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0"
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ø

I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330676

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
û
â
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1331493

inputs

states_0_0

states_0_1>
+lstm_cell_40_matmul_readvariableop_resource:	A
-lstm_cell_40_matmul_1_readvariableop_resource:
;
,lstm_cell_40_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢#lstm_cell_40/BiasAdd/ReadVariableOp¢"lstm_cell_40/MatMul/ReadVariableOp¢$lstm_cell_40/MatMul_1/ReadVariableOp
"lstm_cell_40/MatMul/ReadVariableOpReadVariableOp+lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_40/MatMulMatMulinputs*lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell_40/MatMul_1MatMul
states_0_0,lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_40/addAddV2lstm_cell_40/MatMul:product:0lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_40/BiasAddBiasAddlstm_cell_40/add:z:0+lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :á
lstm_cell_40/splitSplit%lstm_cell_40/split/split_dim:output:0lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splito
lstm_cell_40/SigmoidSigmoidlstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_40/Sigmoid_1Sigmoidlstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell_40/mulMullstm_cell_40/Sigmoid_1:y:0
states_0_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_40/TanhTanhlstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_40/mul_1Mullstm_cell_40/Sigmoid:y:0lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
lstm_cell_40/add_1AddV2lstm_cell_40/mul:z:0lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
lstm_cell_40/Sigmoid_2Sigmoidlstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
lstm_cell_40/Tanh_1Tanhlstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_40/mul_2Mullstm_cell_40/Sigmoid_2:y:0lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentitylstm_cell_40/mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identitylstm_cell_40/mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_2Identitylstm_cell_40/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
NoOpNoOp$^lstm_cell_40/BiasAdd/ReadVariableOp#^lstm_cell_40/MatMul/ReadVariableOp%^lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_40/BiasAdd/ReadVariableOp#lstm_cell_40/BiasAdd/ReadVariableOp2H
"lstm_cell_40/MatMul/ReadVariableOp"lstm_cell_40/MatMul/ReadVariableOp2L
$lstm_cell_40/MatMul_1/ReadVariableOp$lstm_cell_40/MatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1
¾
È
while_cond_1330749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1330749___redundant_placeholder05
1while_while_cond_1330749___redundant_placeholder15
1while_while_cond_1330749___redundant_placeholder25
1while_while_cond_1330749___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


Ô
rnn_12_while_cond_1331104*
&rnn_12_while_rnn_12_while_loop_counter0
,rnn_12_while_rnn_12_while_maximum_iterations
rnn_12_while_placeholder
rnn_12_while_placeholder_1
rnn_12_while_placeholder_2
rnn_12_while_placeholder_3,
(rnn_12_while_less_rnn_12_strided_slice_1C
?rnn_12_while_rnn_12_while_cond_1331104___redundant_placeholder0C
?rnn_12_while_rnn_12_while_cond_1331104___redundant_placeholder1C
?rnn_12_while_rnn_12_while_cond_1331104___redundant_placeholder2C
?rnn_12_while_rnn_12_while_cond_1331104___redundant_placeholder3
rnn_12_while_identity
~
rnn_12/while/LessLessrnn_12_while_placeholder(rnn_12_while_less_rnn_12_strided_slice_1*
T0*
_output_shapes
: Y
rnn_12/while/IdentityIdentityrnn_12/while/Less:z:0*
T0
*
_output_shapes
: "7
rnn_12_while_identityrnn_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
àK
ð
rnn_12_while_body_1331105*
&rnn_12_while_rnn_12_while_loop_counter0
,rnn_12_while_rnn_12_while_maximum_iterations
rnn_12_while_placeholder
rnn_12_while_placeholder_1
rnn_12_while_placeholder_2
rnn_12_while_placeholder_3)
%rnn_12_while_rnn_12_strided_slice_1_0e
arnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor_0b
Ornn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0:	e
Qrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0:
_
Prnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0:	
rnn_12_while_identity
rnn_12_while_identity_1
rnn_12_while_identity_2
rnn_12_while_identity_3
rnn_12_while_identity_4
rnn_12_while_identity_5'
#rnn_12_while_rnn_12_strided_slice_1c
_rnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor`
Mrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	c
Ornn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
]
Nrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	¢Ernn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢Drnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢Frnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp
>rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0rnn_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemarnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor_0rnn_12_while_placeholderGrnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Õ
Drnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpOrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ù
5rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMul7rnn_12/while/TensorArrayV2Read/TensorListGetItem:item:0Lrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
Frnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpQrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0à
7rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulrnn_12_while_placeholder_2Nrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
2rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/addAddV2?rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0Arnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
Ernn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpPrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0û
6rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd6rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add:z:0Mrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ç
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/splitSplitGrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:0?rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split³
6rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
2rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mulMul<rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0rnn_12_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
3rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/TanhTanh=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul:rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:07rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1AddV26rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul:z:08rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
5rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul<rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:09rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
1rnn_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_12_while_placeholder_1rnn_12_while_placeholder8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
rnn_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
rnn_12/while/addAddV2rnn_12_while_placeholderrnn_12/while/add/y:output:0*
T0*
_output_shapes
: V
rnn_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_12/while/add_1AddV2&rnn_12_while_rnn_12_while_loop_counterrnn_12/while/add_1/y:output:0*
T0*
_output_shapes
: n
rnn_12/while/IdentityIdentityrnn_12/while/add_1:z:0^rnn_12/while/NoOp*
T0*
_output_shapes
: 
rnn_12/while/Identity_1Identity,rnn_12_while_rnn_12_while_maximum_iterations^rnn_12/while/NoOp*
T0*
_output_shapes
: n
rnn_12/while/Identity_2Identityrnn_12/while/add:z:0^rnn_12/while/NoOp*
T0*
_output_shapes
: ®
rnn_12/while/Identity_3IdentityArnn_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_12/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
rnn_12/while/Identity_4Identity8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0^rnn_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
rnn_12/while/Identity_5Identity8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0^rnn_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
rnn_12/while/NoOpNoOpF^rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpE^rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpG^rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
rnn_12_while_identityrnn_12/while/Identity:output:0";
rnn_12_while_identity_1 rnn_12/while/Identity_1:output:0";
rnn_12_while_identity_2 rnn_12/while/Identity_2:output:0";
rnn_12_while_identity_3 rnn_12/while/Identity_3:output:0";
rnn_12_while_identity_4 rnn_12/while/Identity_4:output:0";
rnn_12_while_identity_5 rnn_12/while/Identity_5:output:0"L
#rnn_12_while_rnn_12_strided_slice_1%rnn_12_while_rnn_12_strided_slice_1_0"¢
Nrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resourcePrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0"¤
Ornn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceQrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0" 
Mrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceOrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0"Ä
_rnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensorarnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
Ernn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpErnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2
Drnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpDrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
Frnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpFrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù$
¤
while_body_1330314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
$while_stacked_rnn_cells_12_1330338_0:	8
$while_stacked_rnn_cells_12_1330340_0:
3
$while_stacked_rnn_cells_12_1330342_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
"while_stacked_rnn_cells_12_1330338:	6
"while_stacked_rnn_cells_12_1330340:
1
"while_stacked_rnn_cells_12_1330342:	¢2while/stacked_rnn_cells_12/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0å
2while/stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3$while_stacked_rnn_cells_12_1330338_0$while_stacked_rnn_cells_12_1330340_0$while_stacked_rnn_cells_12_1330342_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330220ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp3^while/stacked_rnn_cells_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"J
"while_stacked_rnn_cells_12_1330338$while_stacked_rnn_cells_12_1330338_0"J
"while_stacked_rnn_cells_12_1330340$while_stacked_rnn_cells_12_1330340_0"J
"while_stacked_rnn_cells_12_1330342$while_stacked_rnn_cells_12_1330342_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2h
2while/stacked_rnn_cells_12/StatefulPartitionedCall2while/stacked_rnn_cells_12/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
û
ø
.__inference_lstm_cell_41_layer_call_fn_1332295

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330563p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ò:
É
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330384

inputs/
stacked_rnn_cells_12_1330301:	0
stacked_rnn_cells_12_1330303:
+
stacked_rnn_cells_12_1330305:	
identity

identity_1

identity_2¢,stacked_rnn_cells_12/StatefulPartitionedCall¢while;
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask§
,stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0stacked_rnn_cells_12_1330301stacked_rnn_cells_12_1330303stacked_rnn_cells_12_1330305*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330220n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_12_1330301stacked_rnn_cells_12_1330303stacked_rnn_cells_12_1330305*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1330314*
condR
while_cond_1330313*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
NoOpNoOp-^stacked_rnn_cells_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2\
,stacked_rnn_cells_12/StatefulPartitionedCall,stacked_rnn_cells_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
áU
º
C__inference_rnn_12_layer_call_and_return_conditional_losses_1332161

inputsS
@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	V
Bstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
P
Astacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp¢while;
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¹
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0À
(stacked_rnn_cells_12/lstm_cell_40/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0º
*stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulzeros:output:0Astacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%stacked_rnn_cells_12/lstm_cell_40/addAddV22stacked_rnn_cells_12/lstm_cell_40/MatMul:product:04stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
)stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd)stacked_rnn_cells_12/lstm_cell_40/add:z:0@stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'stacked_rnn_cells_12/lstm_cell_40/splitSplit:stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:02stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
%stacked_rnn_cells_12/lstm_cell_40/mulMul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stacked_rnn_cells_12/lstm_cell_40/TanhTanh0stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
'stacked_rnn_cells_12/lstm_cell_40/mul_1Mul-stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:0*stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'stacked_rnn_cells_12/lstm_cell_40/add_1AddV2)stacked_rnn_cells_12/lstm_cell_40/mul:z:0+stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh+stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
'stacked_rnn_cells_12/lstm_cell_40/mul_2Mul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:0,stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ç
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1332076*
condR
while_cond_1332075*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°	ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp9^stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8^stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:^stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ°	: : : 2t
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2v
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
û
ø
.__inference_lstm_cell_41_layer_call_fn_1332312

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330676p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¾
È
while_cond_1331643
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1331643___redundant_placeholder05
1while_while_cond_1331643___redundant_placeholder15
1while_while_cond_1331643___redundant_placeholder25
1while_while_cond_1331643___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ó
ô
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330593

inputs!
rnn_12_1330518:	"
rnn_12_1330520:

rnn_12_1330522:	'
lstm_cell_41_1330564:	(
lstm_cell_41_1330566:
#
lstm_cell_41_1330568:	#
dense_14_1330584:	
dense_14_1330586:
identity¢ dense_14/StatefulPartitionedCall¢$lstm_cell_41/StatefulPartitionedCall¢rnn_12/StatefulPartitionedCall«
rnn_12/StatefulPartitionedCallStatefulPartitionedCallinputsrnn_12_1330518rnn_12_1330520rnn_12_1330522*
Tin
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_rnn_12_layer_call_and_return_conditional_losses_1330517h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ü
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask­
$lstm_cell_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0'rnn_12/StatefulPartitionedCall:output:1'rnn_12/StatefulPartitionedCall:output:2lstm_cell_41_1330564lstm_cell_41_1330566lstm_cell_41_1330568*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1330563
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-lstm_cell_41/StatefulPartitionedCall:output:0dense_14_1330584dense_14_1330586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_1330583w
stackPack)dense_14/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
	transpose	Transposestack:output:0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall%^lstm_cell_41/StatefulPartitionedCall^rnn_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$lstm_cell_41/StatefulPartitionedCall$lstm_cell_41/StatefulPartitionedCall2@
rnn_12/StatefulPartitionedCallrnn_12/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
é	
É
,__inference_enc_dec_12_layer_call_fn_1330612
input_1
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330593s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
!
_user_specified_name	input_1
àK
ð
rnn_12_while_body_1331284*
&rnn_12_while_rnn_12_while_loop_counter0
,rnn_12_while_rnn_12_while_maximum_iterations
rnn_12_while_placeholder
rnn_12_while_placeholder_1
rnn_12_while_placeholder_2
rnn_12_while_placeholder_3)
%rnn_12_while_rnn_12_strided_slice_1_0e
arnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor_0b
Ornn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0:	e
Qrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0:
_
Prnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0:	
rnn_12_while_identity
rnn_12_while_identity_1
rnn_12_while_identity_2
rnn_12_while_identity_3
rnn_12_while_identity_4
rnn_12_while_identity_5'
#rnn_12_while_rnn_12_strided_slice_1c
_rnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor`
Mrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	c
Ornn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
]
Nrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	¢Ernn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢Drnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢Frnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp
>rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0rnn_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemarnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor_0rnn_12_while_placeholderGrnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Õ
Drnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpOrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ù
5rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMul7rnn_12/while/TensorArrayV2Read/TensorListGetItem:item:0Lrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
Frnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpQrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0à
7rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulrnn_12_while_placeholder_2Nrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
2rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/addAddV2?rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0Arnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
Ernn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpPrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0û
6rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd6rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add:z:0Mrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ç
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/splitSplitGrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:0?rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split³
6rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
2rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mulMul<rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0rnn_12_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
3rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/TanhTanh=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul:rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:07rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1AddV26rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul:z:08rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid=rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
5rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
4rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul<rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:09rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
1rnn_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_12_while_placeholder_1rnn_12_while_placeholder8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
rnn_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
rnn_12/while/addAddV2rnn_12_while_placeholderrnn_12/while/add/y:output:0*
T0*
_output_shapes
: V
rnn_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
rnn_12/while/add_1AddV2&rnn_12_while_rnn_12_while_loop_counterrnn_12/while/add_1/y:output:0*
T0*
_output_shapes
: n
rnn_12/while/IdentityIdentityrnn_12/while/add_1:z:0^rnn_12/while/NoOp*
T0*
_output_shapes
: 
rnn_12/while/Identity_1Identity,rnn_12_while_rnn_12_while_maximum_iterations^rnn_12/while/NoOp*
T0*
_output_shapes
: n
rnn_12/while/Identity_2Identityrnn_12/while/add:z:0^rnn_12/while/NoOp*
T0*
_output_shapes
: ®
rnn_12/while/Identity_3IdentityArnn_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn_12/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
rnn_12/while/Identity_4Identity8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0^rnn_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
rnn_12/while/Identity_5Identity8rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0^rnn_12/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
rnn_12/while/NoOpNoOpF^rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpE^rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpG^rnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
rnn_12_while_identityrnn_12/while/Identity:output:0";
rnn_12_while_identity_1 rnn_12/while/Identity_1:output:0";
rnn_12_while_identity_2 rnn_12/while/Identity_2:output:0";
rnn_12_while_identity_3 rnn_12/while/Identity_3:output:0";
rnn_12_while_identity_4 rnn_12/while/Identity_4:output:0";
rnn_12_while_identity_5 rnn_12/while/Identity_5:output:0"L
#rnn_12_while_rnn_12_strided_slice_1%rnn_12_while_rnn_12_strided_slice_1_0"¢
Nrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resourcePrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0"¤
Ornn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceQrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0" 
Mrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceOrnn_12_while_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0"Ä
_rnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensorarnn_12_while_tensorarrayv2read_tensorlistgetitem_rnn_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
Ernn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpErnn_12/while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2
Drnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpDrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
Frnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpFrnn_12/while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù$
¤
while_body_1330750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
$while_stacked_rnn_cells_12_1330774_0:	8
$while_stacked_rnn_cells_12_1330776_0:
3
$while_stacked_rnn_cells_12_1330778_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
"while_stacked_rnn_cells_12_1330774:	6
"while_stacked_rnn_cells_12_1330776:
1
"while_stacked_rnn_cells_12_1330778:	¢2while/stacked_rnn_cells_12/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0å
2while/stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3$while_stacked_rnn_cells_12_1330774_0$while_stacked_rnn_cells_12_1330776_0$while_stacked_rnn_cells_12_1330778_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330220ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp3^while/stacked_rnn_cells_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"J
"while_stacked_rnn_cells_12_1330774$while_stacked_rnn_cells_12_1330774_0"J
"while_stacked_rnn_cells_12_1330776$while_stacked_rnn_cells_12_1330776_0"J
"while_stacked_rnn_cells_12_1330778$while_stacked_rnn_cells_12_1330778_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2h
2while/stacked_rnn_cells_12/StatefulPartitionedCall2while/stacked_rnn_cells_12/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
æ	
È
,__inference_enc_dec_12_layer_call_fn_1331046

inputs
unknown:	
	unknown_0:

	unknown_1:	
	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330890s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ°	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°	
 
_user_specified_nameinputs
Ù$
¤
while_body_1330066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
$while_stacked_rnn_cells_12_1330090_0:	8
$while_stacked_rnn_cells_12_1330092_0:
3
$while_stacked_rnn_cells_12_1330094_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
"while_stacked_rnn_cells_12_1330090:	6
"while_stacked_rnn_cells_12_1330092:
1
"while_stacked_rnn_cells_12_1330094:	¢2while/stacked_rnn_cells_12/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0å
2while/stacked_rnn_cells_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3$while_stacked_rnn_cells_12_1330090_0$while_stacked_rnn_cells_12_1330092_0$while_stacked_rnn_cells_12_1330094_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330052ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity;while/stacked_rnn_cells_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp3^while/stacked_rnn_cells_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"J
"while_stacked_rnn_cells_12_1330090$while_stacked_rnn_cells_12_1330090_0"J
"while_stacked_rnn_cells_12_1330092$while_stacked_rnn_cells_12_1330092_0"J
"while_stacked_rnn_cells_12_1330094$while_stacked_rnn_cells_12_1330094_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2h
2while/stacked_rnn_cells_12/StatefulPartitionedCall2while/stacked_rnn_cells_12/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ì	
÷
E__inference_dense_14_layer_call_and_return_conditional_losses_1332180

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
È
while_cond_1331931
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1331931___redundant_placeholder05
1while_while_cond_1331931___redundant_placeholder15
1while_while_cond_1331931___redundant_placeholder25
1while_while_cond_1331931___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ø

I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1330039

inputs

states
states_11
matmul_readvariableop_resource:	4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
ÏP

 __inference__traced_save_1332504
file_prefix9
5savev2_enc_dec_12_dense_14_kernel_read_readvariableop7
3savev2_enc_dec_12_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopY
Usavev2_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_read_readvariableopc
_savev2_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_read_readvariableopW
Ssavev2_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_read_readvariableop=
9savev2_enc_dec_12_lstm_cell_41_kernel_read_readvariableopG
Csavev2_enc_dec_12_lstm_cell_41_recurrent_kernel_read_readvariableop;
7savev2_enc_dec_12_lstm_cell_41_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop@
<savev2_adam_enc_dec_12_dense_14_kernel_m_read_readvariableop>
:savev2_adam_enc_dec_12_dense_14_bias_m_read_readvariableop`
\savev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_m_read_readvariableopj
fsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_m_read_readvariableop^
Zsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_m_read_readvariableopD
@savev2_adam_enc_dec_12_lstm_cell_41_kernel_m_read_readvariableopN
Jsavev2_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_m_read_readvariableopB
>savev2_adam_enc_dec_12_lstm_cell_41_bias_m_read_readvariableop@
<savev2_adam_enc_dec_12_dense_14_kernel_v_read_readvariableop>
:savev2_adam_enc_dec_12_dense_14_bias_v_read_readvariableop`
\savev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_v_read_readvariableopj
fsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_v_read_readvariableop^
Zsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_v_read_readvariableopD
@savev2_adam_enc_dec_12_lstm_cell_41_kernel_v_read_readvariableopN
Jsavev2_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_v_read_readvariableopB
>savev2_adam_enc_dec_12_lstm_cell_41_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ³
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ü
valueÒBÏ$B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ù
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_enc_dec_12_dense_14_kernel_read_readvariableop3savev2_enc_dec_12_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopUsavev2_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_read_readvariableop_savev2_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_read_readvariableopSsavev2_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_read_readvariableop9savev2_enc_dec_12_lstm_cell_41_kernel_read_readvariableopCsavev2_enc_dec_12_lstm_cell_41_recurrent_kernel_read_readvariableop7savev2_enc_dec_12_lstm_cell_41_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop<savev2_adam_enc_dec_12_dense_14_kernel_m_read_readvariableop:savev2_adam_enc_dec_12_dense_14_bias_m_read_readvariableop\savev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_m_read_readvariableopfsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_m_read_readvariableopZsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_m_read_readvariableop@savev2_adam_enc_dec_12_lstm_cell_41_kernel_m_read_readvariableopJsavev2_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_m_read_readvariableop>savev2_adam_enc_dec_12_lstm_cell_41_bias_m_read_readvariableop<savev2_adam_enc_dec_12_dense_14_kernel_v_read_readvariableop:savev2_adam_enc_dec_12_dense_14_bias_v_read_readvariableop\savev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_kernel_v_read_readvariableopfsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_recurrent_kernel_v_read_readvariableopZsavev2_adam_enc_dec_12_rnn_12_stacked_rnn_cells_12_lstm_cell_40_bias_v_read_readvariableop@savev2_adam_enc_dec_12_lstm_cell_41_kernel_v_read_readvariableopJsavev2_adam_enc_dec_12_lstm_cell_41_recurrent_kernel_v_read_readvariableop>savev2_adam_enc_dec_12_lstm_cell_41_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*
_input_shapes
: :	:: : : : : :	:
::	:
:: : : : : : :	::	:
::	:
::	::	:
::	:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 
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
:	:&	"
 
_output_shapes
:
:!


_output_shapes	
::%!

_output_shapes
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::
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
:	: 

_output_shapes
::%!

_output_shapes
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:&"
 
_output_shapes
:
:! 

_output_shapes	
::%!!

_output_shapes
:	:&""
 
_output_shapes
:
:!#

_output_shapes	
::$

_output_shapes
: 
ÂC


while_body_1331644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0:	^
Jwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0:
X
Iwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	\
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
V
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	¢>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ç
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOpHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ä
.while/stacked_rnn_cells_12/lstm_cell_40/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0Ewhile/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ë
0while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulwhile_placeholder_2Gwhile/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+while/stacked_rnn_cells_12/lstm_cell_40/addAddV28while/stacked_rnn_cells_12/lstm_cell_40/MatMul:product:0:while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0æ
/while/stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd/while/stacked_rnn_cells_12/lstm_cell_40/add:z:0Fwhile/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7while/stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :²
-while/stacked_rnn_cells_12/lstm_cell_40/splitSplit@while/stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:08while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¥
/while/stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
+while/stacked_rnn_cells_12/lstm_cell_40/mulMul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,while/stacked_rnn_cells_12/lstm_cell_40/TanhTanh6while/stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_1Mul3while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:00while/stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
-while/stacked_rnn_cells_12/lstm_cell_40/add_1AddV2/while/stacked_rnn_cells_12/lstm_cell_40/mul:z:01while/stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
1while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid6while/stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
-while/stacked_rnn_cells_12/lstm_cell_40/mul_2Mul5while/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:02while/stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/stacked_rnn_cells_12/lstm_cell_40/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity1while/stacked_rnn_cells_12/lstm_cell_40/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/NoOpNoOp?^while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>^while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp@^while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"
Gwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resourceIwhile_stacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource_0"
Hwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceJwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource_0"
Fwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceHwhile_stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp>while/stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2~
=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp=while/stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2
?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp?while/stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ä
É
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1330052

inputs

states
states_1'
lstm_cell_40_1330040:	(
lstm_cell_40_1330042:
#
lstm_cell_40_1330044:	
identity

identity_1

identity_2¢$lstm_cell_40/StatefulPartitionedCallÝ
$lstm_cell_40/StatefulPartitionedCallStatefulPartitionedCallinputsstatesstates_1lstm_cell_40_1330040lstm_cell_40_1330042lstm_cell_40_1330044*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1330039}
IdentityIdentity-lstm_cell_40/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Identity_1Identity-lstm_cell_40/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Identity_2Identity-lstm_cell_40/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
NoOpNoOp%^lstm_cell_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_40/StatefulPartitionedCall$lstm_cell_40/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
V
¼
C__inference_rnn_12_layer_call_and_return_conditional_losses_1331873
inputs_0S
@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	V
Bstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
P
Astacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp¢while=
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¹
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0À
(stacked_rnn_cells_12/lstm_cell_40/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0º
*stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulzeros:output:0Astacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%stacked_rnn_cells_12/lstm_cell_40/addAddV22stacked_rnn_cells_12/lstm_cell_40/MatMul:product:04stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
)stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd)stacked_rnn_cells_12/lstm_cell_40/add:z:0@stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'stacked_rnn_cells_12/lstm_cell_40/splitSplit:stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:02stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
%stacked_rnn_cells_12/lstm_cell_40/mulMul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stacked_rnn_cells_12/lstm_cell_40/TanhTanh0stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
'stacked_rnn_cells_12/lstm_cell_40/mul_1Mul-stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:0*stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'stacked_rnn_cells_12/lstm_cell_40/add_1AddV2)stacked_rnn_cells_12/lstm_cell_40/mul:z:0+stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh+stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
'stacked_rnn_cells_12/lstm_cell_40/mul_2Mul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:0,stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ç
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1331788*
condR
while_cond_1331787*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp9^stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8^stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:^stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2t
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2v
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
û
ø
.__inference_lstm_cell_40_layer_call_fn_1332197

inputs
states_0
states_1
unknown:	
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1330039p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
V
¼
C__inference_rnn_12_layer_call_and_return_conditional_losses_1331729
inputs_0S
@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource:	V
Bstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource:
P
Astacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp¢7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp¢9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp¢while=
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
valueB:Ñ
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
B :s
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
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¹
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOpReadVariableOp@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0À
(stacked_rnn_cells_12/lstm_cell_40/MatMulMatMulstrided_slice_2:output:0?stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOpReadVariableOpBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0º
*stacked_rnn_cells_12/lstm_cell_40/MatMul_1MatMulzeros:output:0Astacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
%stacked_rnn_cells_12/lstm_cell_40/addAddV22stacked_rnn_cells_12/lstm_cell_40/MatMul:product:04stacked_rnn_cells_12/lstm_cell_40/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOpReadVariableOpAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
)stacked_rnn_cells_12/lstm_cell_40/BiasAddBiasAdd)stacked_rnn_cells_12/lstm_cell_40/add:z:0@stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1stacked_rnn_cells_12/lstm_cell_40/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'stacked_rnn_cells_12/lstm_cell_40/splitSplit:stacked_rnn_cells_12/lstm_cell_40/split/split_dim:output:02stacked_rnn_cells_12/lstm_cell_40/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)stacked_rnn_cells_12/lstm_cell_40/SigmoidSigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
%stacked_rnn_cells_12/lstm_cell_40/mulMul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stacked_rnn_cells_12/lstm_cell_40/TanhTanh0stacked_rnn_cells_12/lstm_cell_40/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
'stacked_rnn_cells_12/lstm_cell_40/mul_1Mul-stacked_rnn_cells_12/lstm_cell_40/Sigmoid:y:0*stacked_rnn_cells_12/lstm_cell_40/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
'stacked_rnn_cells_12/lstm_cell_40/add_1AddV2)stacked_rnn_cells_12/lstm_cell_40/mul:z:0+stacked_rnn_cells_12/lstm_cell_40/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2Sigmoid0stacked_rnn_cells_12/lstm_cell_40/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(stacked_rnn_cells_12/lstm_cell_40/Tanh_1Tanh+stacked_rnn_cells_12/lstm_cell_40/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
'stacked_rnn_cells_12/lstm_cell_40/mul_2Mul/stacked_rnn_cells_12/lstm_cell_40/Sigmoid_2:y:0,stacked_rnn_cells_12/lstm_cell_40/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ç
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0@stacked_rnn_cells_12_lstm_cell_40_matmul_readvariableop_resourceBstacked_rnn_cells_12_lstm_cell_40_matmul_1_readvariableop_resourceAstacked_rnn_cells_12_lstm_cell_40_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1331644*
condR
while_cond_1331643*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp9^stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8^stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp:^stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2t
8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp8stacked_rnn_cells_12/lstm_cell_40/BiasAdd/ReadVariableOp2r
7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp7stacked_rnn_cells_12/lstm_cell_40/MatMul/ReadVariableOp2v
9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp9stacked_rnn_cells_12/lstm_cell_40/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
@
input_15
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ°	@
output_14
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:´¤
Ö
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
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
	cells
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
'
0"
trackable_list_wrapper
»

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
é
(iter

)beta_1

*beta_2
	+decay
,learning_rate mv!mw-mx.my/mz0m{1m|2m} v~!v-v.v/v0v1v2v"
	optimizer
X
-0
.1
/2
03
14
25
 6
!7"
trackable_list_wrapper
X
-0
.1
/2
03
14
25
 6
!7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
,__inference_enc_dec_12_layer_call_fn_1330612
,__inference_enc_dec_12_layer_call_fn_1331025
,__inference_enc_dec_12_layer_call_fn_1331046
,__inference_enc_dec_12_layer_call_fn_1330930´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1331225
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1331404
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330964
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330998´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
"__inference__wrapped_model_1329967input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
8serving_default"
signature_map
ø
9
state_size

-kernel
.recurrent_kernel
/bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>_random_generator
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ë2È
6__inference_stacked_rnn_cells_12_layer_call_fn_1331444
6__inference_stacked_rnn_cells_12_layer_call_fn_1331461Õ
Ì²È
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2þ
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1331493
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1331525Õ
Ì²È
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
'
F0"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Gstates
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
(__inference_rnn_12_layer_call_fn_1331540
(__inference_rnn_12_layer_call_fn_1331555
(__inference_rnn_12_layer_call_fn_1331570
(__inference_rnn_12_layer_call_fn_1331585æ
Ý²Ù
FullArgSpecO
argsGD
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
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ý
C__inference_rnn_12_layer_call_and_return_conditional_losses_1331729
C__inference_rnn_12_layer_call_and_return_conditional_losses_1331873
C__inference_rnn_12_layer_call_and_return_conditional_losses_1332017
C__inference_rnn_12_layer_call_and_return_conditional_losses_1332161æ
Ý²Ù
FullArgSpecO
argsGD
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
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ø
M
state_size

0kernel
1recurrent_kernel
2bias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R_random_generator
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
-:+	2enc_dec_12/dense_14/kernel
&:$2enc_dec_12/dense_14/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_14_layer_call_fn_1332170¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_14_layer_call_and_return_conditional_losses_1332180¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
M:K	2:enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel
X:V
2Denc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel
G:E28enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias
1:/	2enc_dec_12/lstm_cell_41/kernel
<::
2(enc_dec_12/lstm_cell_41/recurrent_kernel
+:)2enc_dec_12/lstm_cell_41/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
5
Z0
[1
\2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÌBÉ
%__inference_signature_wrapper_1331427input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¤2¡
.__inference_lstm_cell_40_layer_call_fn_1332197
.__inference_lstm_cell_40_layer_call_fn_1332214¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1332246
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1332278¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
'
0"
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
b0"
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
00
11
22"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
¤2¡
.__inference_lstm_cell_41_layer_call_fn_1332295
.__inference_lstm_cell_41_layer_call_fn_1332312¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1332344
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1332376¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
	htotal
	icount
j	variables
k	keras_api"
_tf_keras_metric
^
	ltotal
	mcount
n
_fn_kwargs
o	variables
p	keras_api"
_tf_keras_metric
^
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api"
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
h0
i1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
2:0	2!Adam/enc_dec_12/dense_14/kernel/m
+:)2Adam/enc_dec_12/dense_14/bias/m
R:P	2AAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/m
]:[
2KAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/m
L:J2?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/m
6:4	2%Adam/enc_dec_12/lstm_cell_41/kernel/m
A:?
2/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/m
0:.2#Adam/enc_dec_12/lstm_cell_41/bias/m
2:0	2!Adam/enc_dec_12/dense_14/kernel/v
+:)2Adam/enc_dec_12/dense_14/bias/v
R:P	2AAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/kernel/v
]:[
2KAdam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/recurrent_kernel/v
L:J2?Adam/enc_dec_12/rnn_12/stacked_rnn_cells_12/lstm_cell_40/bias/v
6:4	2%Adam/enc_dec_12/lstm_cell_41/kernel/v
A:?
2/Adam/enc_dec_12/lstm_cell_41/recurrent_kernel/v
0:.2#Adam/enc_dec_12/lstm_cell_41/bias/v 
"__inference__wrapped_model_1329967z-./012 !5¢2
+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ°	
ª "7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿ¦
E__inference_dense_14_layer_call_and_return_conditional_losses_1332180] !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_dense_14_layer_call_fn_1332170P !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ»
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330964p-./012 !9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ°	
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 »
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1330998p-./012 !9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ°	
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 º
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1331225o-./012 !8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ°	
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 º
G__inference_enc_dec_12_layer_call_and_return_conditional_losses_1331404o-./012 !8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ°	
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_enc_dec_12_layer_call_fn_1330612c-./012 !9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ°	
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_enc_dec_12_layer_call_fn_1330930c-./012 !9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ°	
p
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_enc_dec_12_layer_call_fn_1331025b-./012 !8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ°	
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_enc_dec_12_layer_call_fn_1331046b-./012 !8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ°	
p
ª "ÿÿÿÿÿÿÿÿÿÐ
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1332246-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ð
I__inference_lstm_cell_40_layer_call_and_return_conditional_losses_1332278-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¥
.__inference_lstm_cell_40_layer_call_fn_1332197ò-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¥
.__inference_lstm_cell_40_layer_call_fn_1332214ò-./¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÐ
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1332344012¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ð
I__inference_lstm_cell_41_layer_call_and_return_conditional_losses_1332376012¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¥
.__inference_lstm_cell_41_layer_call_fn_1332295ò012¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¥
.__inference_lstm_cell_41_layer_call_fn_1332312ò012¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
MJ
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ
C__inference_rnn_12_layer_call_and_return_conditional_losses_1331729Ò-./S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "v¢s
li

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 
C__inference_rnn_12_layer_call_and_return_conditional_losses_1331873Ò-./S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "v¢s
li

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 
C__inference_rnn_12_layer_call_and_return_conditional_losses_1332017Ã-./D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿ°	

 
p 

 

 
ª "v¢s
li

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 
C__inference_rnn_12_layer_call_and_return_conditional_losses_1332161Ã-./D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿ°	

 
p

 

 
ª "v¢s
li

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ï
(__inference_rnn_12_layer_call_fn_1331540Â-./S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "fc

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿï
(__inference_rnn_12_layer_call_fn_1331555Â-./S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "fc

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿà
(__inference_rnn_12_layer_call_fn_1331570³-./D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿ°	

 
p 

 

 
ª "fc

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿà
(__inference_rnn_12_layer_call_fn_1331585³-./D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿ°	

 
p

 

 
ª "fc

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¯
%__inference_signature_wrapper_1331427-./012 !@¢=
¢ 
6ª3
1
input_1&#
input_1ÿÿÿÿÿÿÿÿÿ°	"7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿñ
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1331493-./¢
¢
 
inputsÿÿÿÿÿÿÿÿÿ
V¢S
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ

 
p 
ª "¢|
u¢r

0/0ÿÿÿÿÿÿÿÿÿ
P¢M
KH
"
0/1/0/0ÿÿÿÿÿÿÿÿÿ
"
0/1/0/1ÿÿÿÿÿÿÿÿÿ
 ñ
Q__inference_stacked_rnn_cells_12_layer_call_and_return_conditional_losses_1331525-./¢
¢
 
inputsÿÿÿÿÿÿÿÿÿ
V¢S
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ

 
p
ª "¢|
u¢r

0/0ÿÿÿÿÿÿÿÿÿ
P¢M
KH
"
0/1/0/0ÿÿÿÿÿÿÿÿÿ
"
0/1/0/1ÿÿÿÿÿÿÿÿÿ
 Æ
6__inference_stacked_rnn_cells_12_layer_call_fn_1331444-./¢
¢
 
inputsÿÿÿÿÿÿÿÿÿ
V¢S
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ

 
p 
ª "o¢l

0ÿÿÿÿÿÿÿÿÿ
L¢I
GD
 
1/0/0ÿÿÿÿÿÿÿÿÿ
 
1/0/1ÿÿÿÿÿÿÿÿÿÆ
6__inference_stacked_rnn_cells_12_layer_call_fn_1331461-./¢
¢
 
inputsÿÿÿÿÿÿÿÿÿ
V¢S
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ

 
p
ª "o¢l

0ÿÿÿÿÿÿÿÿÿ
L¢I
GD
 
1/0/0ÿÿÿÿÿÿÿÿÿ
 
1/0/1ÿÿÿÿÿÿÿÿÿ