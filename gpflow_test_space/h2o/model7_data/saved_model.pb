�	
�&�%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
B
AssignVariableOp
resource
value"dtype"
dtypetype�
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
9
Cholesky

input"T
output"T"
Ttype:	
2
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
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
:
Less
x"T
y"T
z
"
Ttype:
2	
\
ListDiff
x"T
y"T
out"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
.
Log1p
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MatrixDiagV3
diagonal"T
k
num_rows
num_cols
padding_value"T
output"T"	
Ttype"Q
alignstring
RIGHT_LEFT:2
0
LEFT_RIGHT
RIGHT_LEFT	LEFT_LEFTRIGHT_RIGHT
y
MatrixTriangularSolve
matrix"T
rhs"T
output"T"
lowerbool("
adjointbool( "
Ttype:	
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
)
Rank

input"T

output"	
Ttype
@
ReadVariableOp
resource
value"dtype"
dtypetype�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
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
@
Softplus
features"T
activations"T"
Ttype:
2
3
Square
x"T
y"T"
Ttype:
2
	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring �
�
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
h

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
h

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
�

ConstConst*
_output_shapes

:2*
dtype0*�	
value�	B�	2"�	�7#%ۉտPyQ����`2���Y���\@�PyQ��?!�`2���?��o�Y�?�PyQ��?!�`2���?�������PyQ����O[S��&@��o�?��]�^)�@��4�?� �H�����]�^)�@��4�?��@� ��?��]�^)�!?��4��ѷ8,�?�PyQ��?)TX!���?uJ����PyQ���
TX!�����ȳ�ο�PyQ��?)TX!���?��1䃽�PyQ����O[S���⚙��?�PyQ��?!�`2���?ڙh�*��?(PyQ�ݿ��O[Sܿl�z��Z����]�^)�!?��4��J��s���?�PyQ��?)TX!���?��rB�PyQ���
TX!����Ѭ��q��?��]�^)�!?��4����dl�S�?�PyQ��?<�O[S�?�0��ڿ��]�^)�*>��4��$pE� �?�PyQ��?<�O[S�?LK]/��PyQ����`2��z��V�?��]�^)�*>��4��e��S* @��=8�?\D]Ӂ�?i���e�?(PyQ�ݿ��8�K����w���?�PyQ��??�`2���?�����(PyQ�ݿ��O[Sܿh���]y�?�PyQ��??�`2���?��g�pO����=8��"��+���R��j��(PyQ�ݿ��8�K��!�
m���?�PyQ��?!�`2���?k�aҭп�PyQ��?<�O[S�?6S�S�?��=8�?\D]Ӂ�?{���ԡ��PyQ��?N�>���?�Z���?��=8�?\D]Ӂ�?Cd_y?z�?�PyQ��?)TX!���?[$+��u���]�^)�*>��4���e��3���PyQ��??�`2���?l��v�4�?�PyQ��?N�>���?*B�u��?��=8�?\D]Ӂ�?�V(4av���PyQ��?N�>���?c���M�?PyQ���
TX!����14��Jٿ��=8��"��+�������տ��]�^)�@��4�?��#sTC���=8��"��+���rk>�	��(PyQ�ݿ��8�K���ʡ��v�?�PyQ��?N�>���?A�ulK|���PyQ��??�`2���?�k�:<�(PyQ�ݿ��O[SܿJ�ݏۿ��]�^)�!?��4���`@�?�PyQ��?<�O[S�?
�
Const_1Const*
_output_shapes

:2*
dtype0*�
value�B�2"�              �? x"k!p�? ����T�? ��I�?�? �����? `�+i%�? D�x��?  �o��? ��x��? �ʑ�H�? ��x��? ����<�? �"��? �q�ǉ�? [ő��? @�)�? P ��B�? hoM��? ���0�? @60�? ��B���? ��/'r�? ���,�? P5~���? Dk��a�? @娿�? ��pK�? �s$�q�? ����0�? �>��? �8��?  ՛]�? LFY%�? P&��? hN�}�? �[���? �Q��Y�? X���? ���? �W|�z�? ��|b��? ��н��? 0!A��? `�3���? �1���? �����?  �+8�? �����? �����?
P
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2�����ư>

NoOpNoOp
�
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
?
mean_function

kernel

likelihood

signatures
 

kernels

variance
 

0
1
=
	_pretransformed_input

_transform_fn

	_bijector
 
variance
lengthscales

variance
b`
VARIABLE_VALUEVariableDlikelihood/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE


_bijectors
=
_pretransformed_input
_transform_fn
	_bijector
=
_pretransformed_input
_transform_fn
	_bijector
=
_pretransformed_input
_transform_fn
	_bijector

0
1
jh
VARIABLE_VALUE
Variable_1Jkernel/kernels/0/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE
 
nl
VARIABLE_VALUE
Variable_2Nkernel/kernels/0/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE
 
jh
VARIABLE_VALUE
Variable_3Jkernel/kernels/1/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
J
serving_default_XnewPlaceholder*
_output_shapes
:*
dtype0
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_XnewConstConst_1
Variable_2
Variable_1
Variable_3VariableConst_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_116427
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpConst_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_116466
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2
Variable_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_116488�
�

!assert_shapes_1_cond_false_1161947
3assert_shapes_1_cond_identity_assert_shapes_1_shape!
assert_shapes_1_cond_identity�
assert_shapes_1/cond/IdentityIdentity3assert_shapes_1_cond_identity_assert_shapes_1_shape*
T0*#
_output_shapes
:���������2
assert_shapes_1/cond/Identity"G
assert_shapes_1_cond_identity&assert_shapes_1/cond/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
w
assert_shapes_cond_false_1160233
/assert_shapes_cond_identity_assert_shapes_shape
assert_shapes_cond_identity�
assert_shapes/cond/IdentityIdentity/assert_shapes_cond_identity_assert_shapes_shape*
T0*#
_output_shapes
:���������2
assert_shapes/cond/Identity"C
assert_shapes_cond_identity$assert_shapes/cond/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
k
 assert_shapes_1_cond_true_116193$
 assert_shapes_1_cond_placeholder!
assert_shapes_1_cond_identity�
assert_shapes_1/cond/ConstConst*
_output_shapes
:*
dtype0*
valueB:2
assert_shapes_1/cond/Const�
assert_shapes_1/cond/IdentityIdentity#assert_shapes_1/cond/Const:output:0*
T0*
_output_shapes
:2
assert_shapes_1/cond/Identity"G
assert_shapes_1_cond_identity&assert_shapes_1/cond/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
�
#assert_shapes_1_cond_1_false_116209;
7assert_shapes_1_cond_1_identity_assert_shapes_1_shape_1#
assert_shapes_1_cond_1_identity�
assert_shapes_1/cond_1/IdentityIdentity7assert_shapes_1_cond_1_identity_assert_shapes_1_shape_1*
T0*#
_output_shapes
:���������2!
assert_shapes_1/cond_1/Identity"K
assert_shapes_1_cond_1_identity(assert_shapes_1/cond_1/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
e
assert_shapes_cond_true_116022"
assert_shapes_cond_placeholder
assert_shapes_cond_identity~
assert_shapes/cond/ConstConst*
_output_shapes
:*
dtype0*
valueB:2
assert_shapes/cond/Const�
assert_shapes/cond/IdentityIdentity!assert_shapes/cond/Const:output:0*
T0*
_output_shapes
:2
assert_shapes/cond/Identity"C
assert_shapes_cond_identity$assert_shapes/cond/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
�
#assert_shapes_1_cond_2_false_116224;
7assert_shapes_1_cond_2_identity_assert_shapes_1_shape_2#
assert_shapes_1_cond_2_identity�
assert_shapes_1/cond_2/IdentityIdentity7assert_shapes_1_cond_2_identity_assert_shapes_1_shape_2*
T0*#
_output_shapes
:���������2!
assert_shapes_1/cond_2/Identity"K
assert_shapes_1_cond_2_identity(assert_shapes_1/cond_2/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
�
#assert_shapes_1_cond_3_false_116239;
7assert_shapes_1_cond_3_identity_assert_shapes_1_shape_3#
assert_shapes_1_cond_3_identity�
assert_shapes_1/cond_3/IdentityIdentity7assert_shapes_1_cond_3_identity_assert_shapes_1_shape_3*
T0*#
_output_shapes
:���������2!
assert_shapes_1/cond_3/Identity"K
assert_shapes_1_cond_3_identity(assert_shapes_1/cond_3/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
�
__inference__traced_save_116466
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop
savev2_const_3

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�BDlikelihood/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/0/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/0/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/1/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
q
"assert_shapes_1_cond_2_true_116223&
"assert_shapes_1_cond_2_placeholder#
assert_shapes_1_cond_2_identity�
assert_shapes_1/cond_2/ConstConst*
_output_shapes
:*
dtype0*
valueB:2
assert_shapes_1/cond_2/Const�
assert_shapes_1/cond_2/IdentityIdentity%assert_shapes_1/cond_2/Const:output:0*
T0*
_output_shapes
:2!
assert_shapes_1/cond_2/Identity"K
assert_shapes_1_cond_2_identity(assert_shapes_1/cond_2/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
q
"assert_shapes_1_cond_3_true_116238&
"assert_shapes_1_cond_3_placeholder#
assert_shapes_1_cond_3_identity�
assert_shapes_1/cond_3/ConstConst*
_output_shapes
:*
dtype0*
valueB:2
assert_shapes_1/cond_3/Const�
assert_shapes_1/cond_3/IdentityIdentity%assert_shapes_1/cond_3/Const:output:0*
T0*
_output_shapes
:2!
assert_shapes_1/cond_3/Identity"K
assert_shapes_1_cond_3_identity(assert_shapes_1/cond_3/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
k
 assert_shapes_cond_1_true_116038$
 assert_shapes_cond_1_placeholder!
assert_shapes_cond_1_identity�
assert_shapes/cond_1/ConstConst*
_output_shapes
:*
dtype0*
valueB:2
assert_shapes/cond_1/Const�
assert_shapes/cond_1/IdentityIdentity#assert_shapes/cond_1/Const:output:0*
T0*
_output_shapes
:2
assert_shapes/cond_1/Identity"G
assert_shapes_cond_1_identity&assert_shapes/cond_1/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�
�
"__inference__traced_restore_116488
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�BDlikelihood/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/0/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/0/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/1/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
q
"assert_shapes_1_cond_1_true_116208&
"assert_shapes_1_cond_1_placeholder#
assert_shapes_1_cond_1_identity�
assert_shapes_1/cond_1/ConstConst*
_output_shapes
:*
dtype0*
valueB:2
assert_shapes_1/cond_1/Const�
assert_shapes_1/cond_1/IdentityIdentity%assert_shapes_1/cond_1/Const:output:0*
T0*
_output_shapes
:2!
assert_shapes_1/cond_1/Identity"K
assert_shapes_1_cond_1_identity(assert_shapes_1/cond_1/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
�

!assert_shapes_cond_1_false_1160397
3assert_shapes_cond_1_identity_assert_shapes_shape_2!
assert_shapes_cond_1_identity�
assert_shapes/cond_1/IdentityIdentity3assert_shapes_cond_1_identity_assert_shapes_shape_2*
T0*#
_output_shapes
:���������2
assert_shapes/cond_1/Identity"G
assert_shapes_cond_1_identity&assert_shapes/cond_1/Identity:output:0*"
_input_shapes
:���������:) %
#
_output_shapes
:���������
��
�

__inference_predict_f_116404
xnew
shape_115706	
sub_x4
0truediv_softplus_forward_readvariableop_resource,
(softplus_forward_readvariableop_resource4
0squeeze_softplus_forward_readvariableop_resourceE
Afill_3_chain_of_shift_of_softplus_forward_readvariableop_resourceA
=fill_3_chain_of_shift_of_softplus_forward_shift_forward_add_y
identity

identity_1��8Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp�'Squeeze/softplus/forward/ReadVariableOp�)Squeeze_1/softplus/forward/ReadVariableOp�)Squeeze_2/softplus/forward/ReadVariableOp�assert_shapes/Assert/Assert�assert_shapes/Assert_1/Assert�assert_shapes/Assert_2/Assert�assert_shapes/Assert_3/Assert�0assert_shapes/assert_rank_at_least/Assert/Assert�assert_shapes_1/Assert/Assert�assert_shapes_1/Assert_1/Assert�assert_shapes_1/Assert_2/Assert�assert_shapes_1/Assert_3/Assert�assert_shapes_1/Assert_4/Assert�2assert_shapes_1/assert_rank_at_least/Assert/Assert�4assert_shapes_1/assert_rank_at_least_1/Assert/Assert�4assert_shapes_1/assert_rank_at_least_2/Assert/Assert�4assert_shapes_1/assert_rank_at_least_3/Assert/Assert�softplus/forward/ReadVariableOp�!softplus_1/forward/ReadVariableOp�'truediv/softplus/forward/ReadVariableOp�)truediv_1/softplus/forward/ReadVariableOp�)truediv_2/softplus/forward/ReadVariableOp_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"2      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicel
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2strided_slice:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatc
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros/Constf
zerosFillconcat:output:0zeros/Const:output:0*
T0*
_output_shapes

:22
zerosQ
subSubsub_xzeros:output:0*
T0*
_output_shapes

:22
sub
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceshape_115706strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:2*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_1�
'truediv/softplus/forward/ReadVariableOpReadVariableOp0truediv_softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02)
'truediv/softplus/forward/ReadVariableOp�
truediv/softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2!
truediv/softplus/forward/Less/y�
truediv/softplus/forward/LessLess/truediv/softplus/forward/ReadVariableOp:value:0(truediv/softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2
truediv/softplus/forward/Less�
truediv/softplus/forward/ExpExp/truediv/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
truediv/softplus/forward/Exp�
truediv/softplus/forward/Log1pLog1p truediv/softplus/forward/Exp:y:0*
T0*
_output_shapes
: 2 
truediv/softplus/forward/Log1p�
!truediv/softplus/forward/SoftplusSoftplus/truediv/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!truediv/softplus/forward/Softplus�
!truediv/softplus/forward/SelectV2SelectV2!truediv/softplus/forward/Less:z:0"truediv/softplus/forward/Log1p:y:0/truediv/softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2#
!truediv/softplus/forward/SelectV2�
!truediv/softplus/forward/IdentityIdentity*truediv/softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2#
!truediv/softplus/forward/Identity�
"truediv/softplus/forward/IdentityN	IdentityN*truediv/softplus/forward/SelectV2:output:0/truediv/softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115727*
_output_shapes
: : 2$
"truediv/softplus/forward/IdentityN�
truedivRealDivstrided_slice_1:output:0+truediv/softplus/forward/IdentityN:output:0*
T0*
_output_shapes

:22	
truedivP
SquareSquaretruediv:z:0*
T0*
_output_shapes

:22
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indicesw
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2
Sump
MatMulMatMultruediv:z:0truediv:z:0*
T0*
_output_shapes

:22*
transpose_b(2
MatMulW
mul/xConst*
_output_shapes
: *
dtype0*
valueB 2       �2
mul/x\
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:222
mul�
'adjoint/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'adjoint/matrix_transpose/transpose/perm�
"adjoint/matrix_transpose/transpose	TransposeSum:output:00adjoint/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:22$
"adjoint/matrix_transpose/transposer
addAddV2Sum:output:0&adjoint/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:222
addR
add_1AddV2mul:z:0add:z:0*
T0*
_output_shapes

:222
add_1[
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      �2	
mul_1/x[
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*
_output_shapes

:222
mul_1E
ExpExp	mul_1:z:0*
T0*
_output_shapes

:222
Exp�
softplus/forward/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02!
softplus/forward/ReadVariableOp{
softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2
softplus/forward/Less/y�
softplus/forward/LessLess'softplus/forward/ReadVariableOp:value:0 softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2
softplus/forward/Less}
softplus/forward/ExpExp'softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
softplus/forward/Expt
softplus/forward/Log1pLog1psoftplus/forward/Exp:y:0*
T0*
_output_shapes
: 2
softplus/forward/Log1p�
softplus/forward/SoftplusSoftplus'softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
softplus/forward/Softplus�
softplus/forward/SelectV2SelectV2softplus/forward/Less:z:0softplus/forward/Log1p:y:0'softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2
softplus/forward/SelectV2�
softplus/forward/IdentityIdentity"softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2
softplus/forward/Identity�
softplus/forward/IdentityN	IdentityN"softplus/forward/SelectV2:output:0'softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115753*
_output_shapes
: : 2
softplus/forward/IdentityNl
mul_2Mul#softplus/forward/IdentityN:output:0Exp:y:0*
T0*
_output_shapes

:222
mul_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceshape_115706strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_2c
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"2      2	
Shape_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3�
'Squeeze/softplus/forward/ReadVariableOpReadVariableOp0squeeze_softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02)
'Squeeze/softplus/forward/ReadVariableOp�
Squeeze/softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2!
Squeeze/softplus/forward/Less/y�
Squeeze/softplus/forward/LessLess/Squeeze/softplus/forward/ReadVariableOp:value:0(Squeeze/softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2
Squeeze/softplus/forward/Less�
Squeeze/softplus/forward/ExpExp/Squeeze/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
Squeeze/softplus/forward/Exp�
Squeeze/softplus/forward/Log1pLog1p Squeeze/softplus/forward/Exp:y:0*
T0*
_output_shapes
: 2 
Squeeze/softplus/forward/Log1p�
!Squeeze/softplus/forward/SoftplusSoftplus/Squeeze/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!Squeeze/softplus/forward/Softplus�
!Squeeze/softplus/forward/SelectV2SelectV2!Squeeze/softplus/forward/Less:z:0"Squeeze/softplus/forward/Log1p:y:0/Squeeze/softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2#
!Squeeze/softplus/forward/SelectV2�
!Squeeze/softplus/forward/IdentityIdentity*Squeeze/softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2#
!Squeeze/softplus/forward/Identity�
"Squeeze/softplus/forward/IdentityN	IdentityN*Squeeze/softplus/forward/SelectV2:output:0/Squeeze/softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115776*
_output_shapes
: : 2$
"Squeeze/softplus/forward/IdentityNk
SqueezeSqueeze+Squeeze/softplus/forward/IdentityN:output:0*
T0*
_output_shapes
: 2	
Squeezee
FillFillstrided_slice_3:output:0Squeeze:output:0*
T0*
_output_shapes
:22
FillR
diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
diag/ki
diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
diag/num_rowsi
diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
diag/num_colsq
diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB 2        2
diag/padding_value�
diagMatrixDiagV3Fill:output:0diag/k:output:0diag/num_rows:output:0diag/num_cols:output:0diag/padding_value:output:0*
T0*
_output_shapes

:222
diag`
AddNAddN	mul_2:z:0diag:output:0*
N*
T0*
_output_shapes

:222
AddN
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSlicexnewstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_4c
Shape_2Shapestrided_slice_4:output:0*
T0*#
_output_shapes
:���������2	
Shape_2x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack�
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2�
strided_slice_5StridedSliceShape_2:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask2
strided_slice_5�
)Squeeze_1/softplus/forward/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02+
)Squeeze_1/softplus/forward/ReadVariableOp�
!Squeeze_1/softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2#
!Squeeze_1/softplus/forward/Less/y�
Squeeze_1/softplus/forward/LessLess1Squeeze_1/softplus/forward/ReadVariableOp:value:0*Squeeze_1/softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2!
Squeeze_1/softplus/forward/Less�
Squeeze_1/softplus/forward/ExpExp1Squeeze_1/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2 
Squeeze_1/softplus/forward/Exp�
 Squeeze_1/softplus/forward/Log1pLog1p"Squeeze_1/softplus/forward/Exp:y:0*
T0*
_output_shapes
: 2"
 Squeeze_1/softplus/forward/Log1p�
#Squeeze_1/softplus/forward/SoftplusSoftplus1Squeeze_1/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#Squeeze_1/softplus/forward/Softplus�
#Squeeze_1/softplus/forward/SelectV2SelectV2#Squeeze_1/softplus/forward/Less:z:0$Squeeze_1/softplus/forward/Log1p:y:01Squeeze_1/softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2%
#Squeeze_1/softplus/forward/SelectV2�
#Squeeze_1/softplus/forward/IdentityIdentity,Squeeze_1/softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2%
#Squeeze_1/softplus/forward/Identity�
$Squeeze_1/softplus/forward/IdentityN	IdentityN,Squeeze_1/softplus/forward/SelectV2:output:01Squeeze_1/softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115804*
_output_shapes
: : 2&
$Squeeze_1/softplus/forward/IdentityNq
	Squeeze_1Squeeze-Squeeze_1/softplus/forward/IdentityN:output:0*
T0*
_output_shapes
: 2
	Squeeze_1i
Fill_1Fillstrided_slice_5:output:0Squeeze_1:output:0*
T0*
_output_shapes
:2
Fill_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack�
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1�
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2�
strided_slice_6StridedSlicexnewstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_6c
Shape_3Shapestrided_slice_6:output:0*
T0*#
_output_shapes
:���������2	
Shape_3x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/stack�
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2�
strided_slice_7StridedSliceShape_3:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask2
strided_slice_7�
)Squeeze_2/softplus/forward/ReadVariableOpReadVariableOp0squeeze_softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02+
)Squeeze_2/softplus/forward/ReadVariableOp�
!Squeeze_2/softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2#
!Squeeze_2/softplus/forward/Less/y�
Squeeze_2/softplus/forward/LessLess1Squeeze_2/softplus/forward/ReadVariableOp:value:0*Squeeze_2/softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2!
Squeeze_2/softplus/forward/Less�
Squeeze_2/softplus/forward/ExpExp1Squeeze_2/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2 
Squeeze_2/softplus/forward/Exp�
 Squeeze_2/softplus/forward/Log1pLog1p"Squeeze_2/softplus/forward/Exp:y:0*
T0*
_output_shapes
: 2"
 Squeeze_2/softplus/forward/Log1p�
#Squeeze_2/softplus/forward/SoftplusSoftplus1Squeeze_2/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#Squeeze_2/softplus/forward/Softplus�
#Squeeze_2/softplus/forward/SelectV2SelectV2#Squeeze_2/softplus/forward/Less:z:0$Squeeze_2/softplus/forward/Log1p:y:01Squeeze_2/softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2%
#Squeeze_2/softplus/forward/SelectV2�
#Squeeze_2/softplus/forward/IdentityIdentity,Squeeze_2/softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2%
#Squeeze_2/softplus/forward/Identity�
$Squeeze_2/softplus/forward/IdentityN	IdentityN,Squeeze_2/softplus/forward/SelectV2:output:01Squeeze_2/softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115826*
_output_shapes
: : 2&
$Squeeze_2/softplus/forward/IdentityNq
	Squeeze_2Squeeze-Squeeze_2/softplus/forward/IdentityN:output:0*
T0*
_output_shapes
: 2
	Squeeze_2i
Fill_2Fillstrided_slice_7:output:0Squeeze_2:output:0*
T0*
_output_shapes
:2
Fill_2f
AddN_1AddNFill_1:output:0Fill_2:output:0*
N*
T0*
_output_shapes
:2
AddN_1
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack�
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1�
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2�
strided_slice_8StridedSliceshape_115706strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:2*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack�
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1�
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2�
strided_slice_9StridedSlicexnewstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_9�
)truediv_1/softplus/forward/ReadVariableOpReadVariableOp0truediv_softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02+
)truediv_1/softplus/forward/ReadVariableOp�
!truediv_1/softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2#
!truediv_1/softplus/forward/Less/y�
truediv_1/softplus/forward/LessLess1truediv_1/softplus/forward/ReadVariableOp:value:0*truediv_1/softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2!
truediv_1/softplus/forward/Less�
truediv_1/softplus/forward/ExpExp1truediv_1/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2 
truediv_1/softplus/forward/Exp�
 truediv_1/softplus/forward/Log1pLog1p"truediv_1/softplus/forward/Exp:y:0*
T0*
_output_shapes
: 2"
 truediv_1/softplus/forward/Log1p�
#truediv_1/softplus/forward/SoftplusSoftplus1truediv_1/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#truediv_1/softplus/forward/Softplus�
#truediv_1/softplus/forward/SelectV2SelectV2#truediv_1/softplus/forward/Less:z:0$truediv_1/softplus/forward/Log1p:y:01truediv_1/softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2%
#truediv_1/softplus/forward/SelectV2�
#truediv_1/softplus/forward/IdentityIdentity,truediv_1/softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2%
#truediv_1/softplus/forward/Identity�
$truediv_1/softplus/forward/IdentityN	IdentityN,truediv_1/softplus/forward/SelectV2:output:01truediv_1/softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115849*
_output_shapes
: : 2&
$truediv_1/softplus/forward/IdentityN�
	truediv_1RealDivstrided_slice_8:output:0-truediv_1/softplus/forward/IdentityN:output:0*
T0*
_output_shapes

:22
	truediv_1�
)truediv_2/softplus/forward/ReadVariableOpReadVariableOp0truediv_softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02+
)truediv_2/softplus/forward/ReadVariableOp�
!truediv_2/softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2#
!truediv_2/softplus/forward/Less/y�
truediv_2/softplus/forward/LessLess1truediv_2/softplus/forward/ReadVariableOp:value:0*truediv_2/softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2!
truediv_2/softplus/forward/Less�
truediv_2/softplus/forward/ExpExp1truediv_2/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2 
truediv_2/softplus/forward/Exp�
 truediv_2/softplus/forward/Log1pLog1p"truediv_2/softplus/forward/Exp:y:0*
T0*
_output_shapes
: 2"
 truediv_2/softplus/forward/Log1p�
#truediv_2/softplus/forward/SoftplusSoftplus1truediv_2/softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#truediv_2/softplus/forward/Softplus�
#truediv_2/softplus/forward/SelectV2SelectV2#truediv_2/softplus/forward/Less:z:0$truediv_2/softplus/forward/Log1p:y:01truediv_2/softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2%
#truediv_2/softplus/forward/SelectV2�
#truediv_2/softplus/forward/IdentityIdentity,truediv_2/softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2%
#truediv_2/softplus/forward/Identity�
$truediv_2/softplus/forward/IdentityN	IdentityN,truediv_2/softplus/forward/SelectV2:output:01truediv_2/softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115861*
_output_shapes
: : 2&
$truediv_2/softplus/forward/IdentityN�
	truediv_2RealDivstrided_slice_9:output:0-truediv_2/softplus/forward/IdentityN:output:0*
T0*
_output_shapes
:2
	truediv_2V
Square_1Squaretruediv_1:z:0*
T0*
_output_shapes

:22

Square_1}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum_1/reduction_indicesj
Sum_1SumSquare_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:22
Sum_1P
Square_2Squaretruediv_2:z:0*
T0*
_output_shapes
:2

Square_2}
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum_2/reduction_indicesh
Sum_2SumSquare_2:y:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_2h
Tensordot/ShapeShapetruediv_2:z:0*
T0*#
_output_shapes
:���������2
Tensordot/ShapeX
Tensordot/RankRanktruediv_2:z:0*
T0*
_output_shapes
: 2
Tensordot/Ranks
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
���������2
Tensordot/axesv
Tensordot/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GreaterEqual/y�
Tensordot/GreaterEqualGreaterEqualTensordot/axes:output:0!Tensordot/GreaterEqual/y:output:0*
T0*
_output_shapes
:2
Tensordot/GreaterEqual~
Tensordot/addAddV2Tensordot/axes:output:0Tensordot/Rank:output:0*
T0*
_output_shapes
:2
Tensordot/add�
Tensordot/SelectSelectTensordot/GreaterEqual:z:0Tensordot/axes:output:0Tensordot/add:z:0*
T0*
_output_shapes
:2
Tensordot/Selectp
Tensordot/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/range/startp
Tensordot/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
Tensordot/range/delta�
Tensordot/rangeRangeTensordot/range/start:output:0Tensordot/Rank:output:0Tensordot/range/delta:output:0*#
_output_shapes
:���������2
Tensordot/range�
Tensordot/ListDiffListDiffTensordot/range:output:0Tensordot/Select:output:0*
T0*2
_output_shapes 
:���������:���������2
Tensordot/ListDifft
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/ListDiff:out:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:���������2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/Select:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/Select:output:0Tensordot/ListDiff:out:0Tensordot/concat/axis:output:0*
N*
T0*#
_output_shapes
:���������2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod_1:output:0Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	Transposetruediv_2:z:0Tensordot/concat:output:0*
T0*
_output_shapes
:2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMultruediv_1:z:0Tensordot/Reshape:output:0*
T0*'
_output_shapes
:2���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/Const_2:output:0Tensordot/GatherV2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*#
_output_shapes
:���������2
Tensordot/concat_1}
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*
_output_shapes
:2
	Tensordot[
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB 2       �2	
mul_3/x^
mul_3Mulmul_3/x:output:0Tensordot:output:0*
T0*
_output_shapes
:2
mul_3o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
Reshape/shapen
ReshapeReshapeSum_1:output:0Reshape/shape:output:0*
T0*
_output_shapes

:22	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����2
Reshape_1/shape}
	Reshape_1ReshapeSum_2:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:���������2
	Reshape_1m
Add_2AddReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:2���������2
Add_2\
Shape_4Const*
_output_shapes
:*
dtype0*
valueB:22	
Shape_4Y
Shape_5ShapeSum_2:output:0*
T0*#
_output_shapes
:���������2	
Shape_5`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis�
concat_1ConcatV2Shape_4:output:0Shape_5:output:0concat_1/axis:output:0*
N*
T0*#
_output_shapes
:���������2

concat_1b
	Reshape_2Reshape	Add_2:z:0concat_1:output:0*
T0*
_output_shapes
:2
	Reshape_2Y
add_3AddV2	mul_3:z:0Reshape_2:output:0*
T0*
_output_shapes
:2
add_3[
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB 2      �2	
mul_4/xU
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes
:2
mul_4C
Exp_1Exp	mul_4:z:0*
T0*
_output_shapes
:2
Exp_1�
!softplus_1/forward/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02#
!softplus_1/forward/ReadVariableOp
softplus_1/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2
softplus_1/forward/Less/y�
softplus_1/forward/LessLess)softplus_1/forward/ReadVariableOp:value:0"softplus_1/forward/Less/y:output:0*
T0*
_output_shapes
: 2
softplus_1/forward/Less�
softplus_1/forward/ExpExp)softplus_1/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
softplus_1/forward/Expz
softplus_1/forward/Log1pLog1psoftplus_1/forward/Exp:y:0*
T0*
_output_shapes
: 2
softplus_1/forward/Log1p�
softplus_1/forward/SoftplusSoftplus)softplus_1/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
softplus_1/forward/Softplus�
softplus_1/forward/SelectV2SelectV2softplus_1/forward/Less:z:0softplus_1/forward/Log1p:y:0)softplus_1/forward/Softplus:activations:0*
T0*
_output_shapes
: 2
softplus_1/forward/SelectV2�
softplus_1/forward/IdentityIdentity$softplus_1/forward/SelectV2:output:0*
T0*
_output_shapes
: 2
softplus_1/forward/Identity�
softplus_1/forward/IdentityN	IdentityN$softplus_1/forward/SelectV2:output:0)softplus_1/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115925*
_output_shapes
: : 2
softplus_1/forward/IdentityNj
mul_5Mul%softplus_1/forward/IdentityN:output:0	Exp_1:y:0*
T0*
_output_shapes
:2
mul_5�
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack�
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1�
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2�
strided_slice_10StridedSliceshape_115706strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:2*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_10�
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack�
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1�
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2�
strided_slice_11StridedSlicexnewstrided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
ellipsis_mask*
end_mask2
strided_slice_11c
Shape_6Const*
_output_shapes
:*
dtype0*
valueB"2      2	
Shape_6z
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/stack�
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2�
strided_slice_12StridedSliceShape_6:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_12d
Shape_7Shapestrided_slice_11:output:0*
T0*#
_output_shapes
:���������2	
Shape_7z
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_13/stack�
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_13/stack_1~
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_2�
strided_slice_13StridedSliceShape_7:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask2
strided_slice_13`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis�
concat_2ConcatV2strided_slice_12:output:0strided_slice_13:output:0concat_2/axis:output:0*
N*
T0*#
_output_shapes
:���������2

concat_2g
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_1/Consth
zeros_1Fillconcat_2:output:0zeros_1/Const:output:0*
T0*
_output_shapes
:2	
zeros_1a
AddN_2AddN	mul_5:z:0zeros_1:output:0*
N*
T0*
_output_shapes
:2
AddN_2d
Fill_3/dimsConst*
_output_shapes
:*
dtype0*
valueB:22
Fill_3/dims�
8Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOpReadVariableOpAfill_3_chain_of_shift_of_softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype02:
8Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp�
AFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB 2      4�2C
AFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Less/y�
?Fill_3/chain_of_shift_of_softplus/forward/softplus/forward/LessLess@Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp:value:0JFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Less/y:output:0*
T0*
_output_shapes
: 2A
?Fill_3/chain_of_shift_of_softplus/forward/softplus/forward/Less�
>Fill_3/chain_of_shift_of_softplus/forward/softplus/forward/ExpExp@Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2@
>Fill_3/chain_of_shift_of_softplus/forward/softplus/forward/Exp�
@Fill_3/chain_of_shift_of_softplus/forward/softplus/forward/Log1pLog1pBFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Exp:y:0*
T0*
_output_shapes
: 2B
@Fill_3/chain_of_shift_of_softplus/forward/softplus/forward/Log1p�
CFill_3/chain_of_shift_of_softplus/forward/softplus/forward/SoftplusSoftplus@Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: 2E
CFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Softplus�
CFill_3/chain_of_shift_of_softplus/forward/softplus/forward/SelectV2SelectV2CFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Less:z:0DFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Log1p:y:0QFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: 2E
CFill_3/chain_of_shift_of_softplus/forward/softplus/forward/SelectV2�
CFill_3/chain_of_shift_of_softplus/forward/softplus/forward/IdentityIdentityLFill_3/chain_of_shift_of_softplus/forward/softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: 2E
CFill_3/chain_of_shift_of_softplus/forward/softplus/forward/Identity�
DFill_3/chain_of_shift_of_softplus/forward/softplus/forward/IdentityN	IdentityNLFill_3/chain_of_shift_of_softplus/forward/softplus/forward/SelectV2:output:0@Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-115963*
_output_shapes
: : 2F
DFill_3/chain_of_shift_of_softplus/forward/softplus/forward/IdentityN�
;Fill_3/chain_of_shift_of_softplus/forward/shift/forward/addAddV2MFill_3/chain_of_shift_of_softplus/forward/softplus/forward/IdentityN:output:0=fill_3_chain_of_shift_of_softplus_forward_shift_forward_add_y*
T0*
_output_shapes
: 2=
;Fill_3/chain_of_shift_of_softplus/forward/shift/forward/add�
Fill_3FillFill_3/dims:output:0?Fill_3/chain_of_shift_of_softplus/forward/shift/forward/add:z:0*
T0*
_output_shapes
:22
Fill_3V
diag_1/kConst*
_output_shapes
: *
dtype0*
value	B : 2

diag_1/km
diag_1/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
diag_1/num_rowsm
diag_1/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
diag_1/num_colsu
diag_1/padding_valueConst*
_output_shapes
: *
dtype0*
valueB 2        2
diag_1/padding_value�
diag_1MatrixDiagV3Fill_3:output:0diag_1/k:output:0diag_1/num_rows:output:0diag_1/num_cols:output:0diag_1/padding_value:output:0*
T0*
_output_shapes

:222
diag_1]
add_4AddV2
AddN:sum:0diag_1:output:0*
T0*
_output_shapes

:222
add_4T
CholeskyCholesky	add_4:z:0*
T0*
_output_shapes

:222

Choleskyc
Shape_8Const*
_output_shapes
:*
dtype0*
valueB"2      2	
Shape_8�
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2�
strided_slice_14StridedSliceShape_8:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_14W
Shape_9ShapeAddN_2:sum:0*
T0*#
_output_shapes
:���������2	
Shape_9�
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_15/stack~
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_15/stack_1~
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_2�
strided_slice_15StridedSliceShape_9:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_15e
Shape_10Const*
_output_shapes
:*
dtype0*
valueB"2      2

Shape_10�
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_16/stack�
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_16/stack_1~
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_2�
strided_slice_16StridedSliceShape_10:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_16C
RankRankAddN_2:sum:0*
T0*
_output_shapes
: 2
RankT
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
sub_1/yW
sub_1SubRank:output:0sub_1/y:output:0*
T0*
_output_shapes
: 2
sub_1\
range/startConst*
_output_shapes
: *
dtype0*
value	B :2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltas
rangeRangerange/start:output:0	sub_1:z:0range/delta:output:0*#
_output_shapes
:���������2
rangeT
sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
sub_2/yW
sub_2SubRank:output:0sub_2/y:output:0*
T0*
_output_shapes
: 2
sub_2c
Reshape_3/shapePack	sub_2:z:0*
N*
T0*
_output_shapes
:2
Reshape_3/shapey
	Reshape_3Reshaperange:output:0Reshape_3/shape:output:0*
T0*#
_output_shapes
:���������2
	Reshape_3f
Reshape_4/tensorConst*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_4/tensorl
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:2
Reshape_4/shape{
	Reshape_4ReshapeReshape_4/tensor:output:0Reshape_4/shape:output:0*
T0*
_output_shapes
:2
	Reshape_4T
sub_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
sub_3/yW
sub_3SubRank:output:0sub_3/y:output:0*
T0*
_output_shapes
: 2
sub_3l
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:2
Reshape_5/shapek
	Reshape_5Reshape	sub_3:z:0Reshape_5/shape:output:0*
T0*
_output_shapes
:2
	Reshape_5`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis�
concat_3ConcatV2Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0concat_3/axis:output:0*
N*
T0*#
_output_shapes
:���������2

concat_3g
	transpose	TransposeAddN_2:sum:0concat_3:output:0*
T0*
_output_shapes
:2
	transposep
assert_shapes/ShapeShapetranspose:y:0*
T0*#
_output_shapes
:���������2
assert_shapes/Shape`
assert_shapes/RankRanktranspose:y:0*
T0*
_output_shapes
: 2
assert_shapes/Rankp
assert_shapes/Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_shapes/Equal/y�
assert_shapes/EqualEqualassert_shapes/Rank:output:0assert_shapes/Equal/y:output:0*
T0*
_output_shapes
: 2
assert_shapes/Equal�
assert_shapes/condStatelessIfassert_shapes/Equal:z:0assert_shapes/Shape:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
else_branch#R!
assert_shapes_cond_false_116023*"
output_shapes
:���������*1
then_branch"R 
assert_shapes_cond_true_1160222
assert_shapes/cond�
assert_shapes/cond/IdentityIdentityassert_shapes/cond:output:0*
T0*#
_output_shapes
:���������2
assert_shapes/cond/Identity
assert_shapes/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"2   2   2
assert_shapes/Shape_1s
assert_shapes/Shape_2ShapeAddN_1:sum:0*
T0*#
_output_shapes
:���������2
assert_shapes/Shape_2c
assert_shapes/Rank_1RankAddN_1:sum:0*
T0*
_output_shapes
: 2
assert_shapes/Rank_1t
assert_shapes/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_shapes/Equal_1/y�
assert_shapes/Equal_1Equalassert_shapes/Rank_1:output:0 assert_shapes/Equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_shapes/Equal_1�
assert_shapes/cond_1StatelessIfassert_shapes/Equal_1:z:0assert_shapes/Shape_2:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
else_branch%R#
!assert_shapes_cond_1_false_116039*"
output_shapes
:���������*3
then_branch$R"
 assert_shapes_cond_1_true_1160382
assert_shapes/cond_1�
assert_shapes/cond_1/IdentityIdentityassert_shapes/cond_1:output:0*
T0*#
_output_shapes
:���������2
assert_shapes/cond_1/Identity
assert_shapes/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"2      2
assert_shapes/Shape_3�
'assert_shapes/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :2)
'assert_shapes/assert_rank_at_least/rank�
(assert_shapes/assert_rank_at_least/ShapeShapetranspose:y:0*
T0*#
_output_shapes
:���������2*
(assert_shapes/assert_rank_at_least/Shape�
Qassert_shapes/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2S
Qassert_shapes/assert_rank_at_least/assert_type/statically_determined_correct_type�
)assert_shapes/assert_rank_at_least/Rank_1Ranktranspose:y:0*
T0*
_output_shapes
: 2+
)assert_shapes/assert_rank_at_least/Rank_1�
/assert_shapes/assert_rank_at_least/GreaterEqualGreaterEqual2assert_shapes/assert_rank_at_least/Rank_1:output:00assert_shapes/assert_rank_at_least/rank:output:0*
T0*
_output_shapes
: 21
/assert_shapes/assert_rank_at_least/GreaterEqual�
/assert_shapes/assert_rank_at_least/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]21
/assert_shapes/assert_rank_at_least/Assert/Const�
1assert_shapes/assert_rank_at_least/Assert/Const_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Tensor transpose:0 must have rank at least23
1assert_shapes/assert_rank_at_least/Assert/Const_1�
1assert_shapes/assert_rank_at_least/Assert/Const_2Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 23
1assert_shapes/assert_rank_at_least/Assert/Const_2�
7assert_shapes/assert_rank_at_least/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]29
7assert_shapes/assert_rank_at_least/Assert/Assert/data_0�
7assert_shapes/assert_rank_at_least/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Tensor transpose:0 must have rank at least29
7assert_shapes/assert_rank_at_least/Assert/Assert/data_1�
7assert_shapes/assert_rank_at_least/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 29
7assert_shapes/assert_rank_at_least/Assert/Assert/data_3�
0assert_shapes/assert_rank_at_least/Assert/AssertAssert3assert_shapes/assert_rank_at_least/GreaterEqual:z:0@assert_shapes/assert_rank_at_least/Assert/Assert/data_0:output:0@assert_shapes/assert_rank_at_least/Assert/Assert/data_1:output:00assert_shapes/assert_rank_at_least/rank:output:0@assert_shapes/assert_rank_at_least/Assert/Assert/data_3:output:01assert_shapes/assert_rank_at_least/Shape:output:0*
T	
2*
_output_shapes
 22
0assert_shapes/assert_rank_at_least/Assert/Assert�
assert_shapes/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :2 
assert_shapes/assert_rank/rank�
assert_shapes/assert_rank/ShapeConst*
_output_shapes
:*
dtype0*
valueB"2   2   2!
assert_shapes/assert_rank/Shape�
Hassert_shapes/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2J
Hassert_shapes/assert_rank/assert_type/statically_determined_correct_type�
9assert_shapes/assert_rank/static_checks_determined_all_okNoOp*
_output_shapes
 2;
9assert_shapes/assert_rank/static_checks_determined_all_ok�
 assert_shapes/assert_rank_1/rankConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_shapes/assert_rank_1/rank�
!assert_shapes/assert_rank_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"2      2#
!assert_shapes/assert_rank_1/Shape�
Jassert_shapes/assert_rank_1/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2L
Jassert_shapes/assert_rank_1/assert_type/statically_determined_correct_type�
;assert_shapes/assert_rank_1/static_checks_determined_all_okNoOp*
_output_shapes
 2=
;assert_shapes/assert_rank_1/static_checks_determined_all_ok�
!assert_shapes/strided_slice/stackConst:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2#
!assert_shapes/strided_slice/stack�
#assert_shapes/strided_slice/stack_1Const:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2%
#assert_shapes/strided_slice/stack_1�
#assert_shapes/strided_slice/stack_2Const:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2%
#assert_shapes/strided_slice/stack_2�
assert_shapes/strided_sliceStridedSlice$assert_shapes/cond/Identity:output:0*assert_shapes/strided_slice/stack:output:0,assert_shapes/strided_slice/stack_1:output:0,assert_shapes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
assert_shapes/strided_slice�
#assert_shapes/strided_slice_1/stackConst:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2%
#assert_shapes/strided_slice_1/stack�
%assert_shapes/strided_slice_1/stack_1Const:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2'
%assert_shapes/strided_slice_1/stack_1�
%assert_shapes/strided_slice_1/stack_2Const:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2'
%assert_shapes/strided_slice_1/stack_2�
assert_shapes/strided_slice_1StridedSlice$assert_shapes/cond/Identity:output:0,assert_shapes/strided_slice_1/stack:output:0.assert_shapes/strided_slice_1/stack_1:output:0.assert_shapes/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
assert_shapes/strided_slice_1l
assert_shapes/ConstConst*
_output_shapes
: *
dtype0*
value	B :22
assert_shapes/Const�
assert_shapes/Equal_2Equalassert_shapes/Const:output:0$assert_shapes/strided_slice:output:0*
T0*
_output_shapes
: 2
assert_shapes/Equal_2
assert_shapes/Shape_4Const*
_output_shapes
:*
dtype0*
valueB"2   2   2
assert_shapes/Shape_4�
assert_shapes/Assert/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2
assert_shapes/Assert/Const�
assert_shapes/Assert/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22
assert_shapes/Assert/Const_1�
assert_shapes/Assert/Const_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor Cholesky:0 dimension2
assert_shapes/Assert/Const_2~
assert_shapes/Assert/Const_3Const*
_output_shapes
: *
dtype0*
value	B : 2
assert_shapes/Assert/Const_3�
assert_shapes/Assert/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2
assert_shapes/Assert/Const_4�
assert_shapes/Assert/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2
assert_shapes/Assert/Const_5�
"assert_shapes/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2$
"assert_shapes/Assert/Assert/data_0�
"assert_shapes/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22$
"assert_shapes/Assert/Assert/data_1�
"assert_shapes/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor Cholesky:0 dimension2$
"assert_shapes/Assert/Assert/data_2�
"assert_shapes/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"assert_shapes/Assert/Assert/data_3�
"assert_shapes/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2$
"assert_shapes/Assert/Assert/data_4�
"assert_shapes/Assert/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2$
"assert_shapes/Assert/Assert/data_6�
assert_shapes/Assert/AssertAssertassert_shapes/Equal_2:z:0+assert_shapes/Assert/Assert/data_0:output:0+assert_shapes/Assert/Assert/data_1:output:0+assert_shapes/Assert/Assert/data_2:output:0+assert_shapes/Assert/Assert/data_3:output:0+assert_shapes/Assert/Assert/data_4:output:0$assert_shapes/strided_slice:output:0+assert_shapes/Assert/Assert/data_6:output:0assert_shapes/Shape_4:output:01^assert_shapes/assert_rank_at_least/Assert/Assert*
T

2*
_output_shapes
 2
assert_shapes/Assert/Assertp
assert_shapes/Const_1Const*
_output_shapes
: *
dtype0*
value	B :22
assert_shapes/Const_1�
assert_shapes/Equal_3Equalassert_shapes/Const_1:output:0$assert_shapes/strided_slice:output:0*
T0*
_output_shapes
: 2
assert_shapes/Equal_3
assert_shapes/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"2   2   2
assert_shapes/Shape_5�
assert_shapes/Assert_1/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2
assert_shapes/Assert_1/Const�
assert_shapes/Assert_1/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22 
assert_shapes/Assert_1/Const_1�
assert_shapes/Assert_1/Const_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor Cholesky:0 dimension2 
assert_shapes/Assert_1/Const_2�
assert_shapes/Assert_1/Const_3Const*
_output_shapes
: *
dtype0*
value	B :2 
assert_shapes/Assert_1/Const_3�
assert_shapes/Assert_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2 
assert_shapes/Assert_1/Const_4�
assert_shapes/Assert_1/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2 
assert_shapes/Assert_1/Const_5�
$assert_shapes/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2&
$assert_shapes/Assert_1/Assert/data_0�
$assert_shapes/Assert_1/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22&
$assert_shapes/Assert_1/Assert/data_1�
$assert_shapes/Assert_1/Assert/data_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor Cholesky:0 dimension2&
$assert_shapes/Assert_1/Assert/data_2�
$assert_shapes/Assert_1/Assert/data_3Const*
_output_shapes
: *
dtype0*
value	B :2&
$assert_shapes/Assert_1/Assert/data_3�
$assert_shapes/Assert_1/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2&
$assert_shapes/Assert_1/Assert/data_4�
$assert_shapes/Assert_1/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2&
$assert_shapes/Assert_1/Assert/data_6�
assert_shapes/Assert_1/AssertAssertassert_shapes/Equal_3:z:0-assert_shapes/Assert_1/Assert/data_0:output:0-assert_shapes/Assert_1/Assert/data_1:output:0-assert_shapes/Assert_1/Assert/data_2:output:0-assert_shapes/Assert_1/Assert/data_3:output:0-assert_shapes/Assert_1/Assert/data_4:output:0$assert_shapes/strided_slice:output:0-assert_shapes/Assert_1/Assert/data_6:output:0assert_shapes/Shape_5:output:0^assert_shapes/Assert/Assert*
T

2*
_output_shapes
 2
assert_shapes/Assert_1/Assert�
#assert_shapes/strided_slice_2/stackConst:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2%
#assert_shapes/strided_slice_2/stack�
%assert_shapes/strided_slice_2/stack_1Const:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2'
%assert_shapes/strided_slice_2/stack_1�
%assert_shapes/strided_slice_2/stack_2Const:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2'
%assert_shapes/strided_slice_2/stack_2�
assert_shapes/strided_slice_2StridedSlice&assert_shapes/cond_1/Identity:output:0,assert_shapes/strided_slice_2/stack:output:0.assert_shapes/strided_slice_2/stack_1:output:0.assert_shapes/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
assert_shapes/strided_slice_2�
assert_shapes/Equal_4Equal&assert_shapes/strided_slice_2:output:0&assert_shapes/strided_slice_1:output:0*
T0*
_output_shapes
: 2
assert_shapes/Equal_4s
assert_shapes/Shape_6ShapeAddN_1:sum:0*
T0*#
_output_shapes
:���������2
assert_shapes/Shape_6�
assert_shapes/Assert_2/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2
assert_shapes/Assert_2/Const�
assert_shapes/Assert_2/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -12 
assert_shapes/Assert_2/Const_1�
assert_shapes/Assert_2/Const_2Const*
_output_shapes
: *
dtype0**
value!B BTensor AddN_1:0 dimension2 
assert_shapes/Assert_2/Const_2�
assert_shapes/Assert_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2 
assert_shapes/Assert_2/Const_3�
assert_shapes/Assert_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2 
assert_shapes/Assert_2/Const_4�
assert_shapes/Assert_2/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2 
assert_shapes/Assert_2/Const_5�
$assert_shapes/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2&
$assert_shapes/Assert_2/Assert/data_0�
$assert_shapes/Assert_2/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -12&
$assert_shapes/Assert_2/Assert/data_1�
$assert_shapes/Assert_2/Assert/data_2Const*
_output_shapes
: *
dtype0**
value!B BTensor AddN_1:0 dimension2&
$assert_shapes/Assert_2/Assert/data_2�
$assert_shapes/Assert_2/Assert/data_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2&
$assert_shapes/Assert_2/Assert/data_3�
$assert_shapes/Assert_2/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2&
$assert_shapes/Assert_2/Assert/data_4�
$assert_shapes/Assert_2/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2&
$assert_shapes/Assert_2/Assert/data_6�
assert_shapes/Assert_2/AssertAssertassert_shapes/Equal_4:z:0-assert_shapes/Assert_2/Assert/data_0:output:0-assert_shapes/Assert_2/Assert/data_1:output:0-assert_shapes/Assert_2/Assert/data_2:output:0-assert_shapes/Assert_2/Assert/data_3:output:0-assert_shapes/Assert_2/Assert/data_4:output:0&assert_shapes/strided_slice_1:output:0-assert_shapes/Assert_2/Assert/data_6:output:0assert_shapes/Shape_6:output:0^assert_shapes/Assert_1/Assert*
T

2*
_output_shapes
 2
assert_shapes/Assert_2/Assertp
assert_shapes/Const_2Const*
_output_shapes
: *
dtype0*
value	B :22
assert_shapes/Const_2�
assert_shapes/Equal_5Equalassert_shapes/Const_2:output:0$assert_shapes/strided_slice:output:0*
T0*
_output_shapes
: 2
assert_shapes/Equal_5
assert_shapes/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"2      2
assert_shapes/Shape_7�
assert_shapes/Assert_3/ConstConst*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2
assert_shapes/Assert_3/Const�
assert_shapes/Assert_3/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22 
assert_shapes/Assert_3/Const_1�
assert_shapes/Assert_3/Const_2Const*
_output_shapes
: *
dtype0*'
valueB BTensor sub:0 dimension2 
assert_shapes/Assert_3/Const_2�
assert_shapes/Assert_3/Const_3Const*
_output_shapes
: *
dtype0*
value	B : 2 
assert_shapes/Assert_3/Const_3�
assert_shapes/Assert_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2 
assert_shapes/Assert_3/Const_4�
assert_shapes/Assert_3/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2 
assert_shapes/Assert_3/Const_5�
$assert_shapes/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*�
value�B� B�base_conditional() arguments [Note that this check verifies the shape of an alternative representation of Kmn. See the docs for the actual expected shape.]2&
$assert_shapes/Assert_3/Assert/data_0�
$assert_shapes/Assert_3/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22&
$assert_shapes/Assert_3/Assert/data_1�
$assert_shapes/Assert_3/Assert/data_2Const*
_output_shapes
: *
dtype0*'
valueB BTensor sub:0 dimension2&
$assert_shapes/Assert_3/Assert/data_2�
$assert_shapes/Assert_3/Assert/data_3Const*
_output_shapes
: *
dtype0*
value	B : 2&
$assert_shapes/Assert_3/Assert/data_3�
$assert_shapes/Assert_3/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2&
$assert_shapes/Assert_3/Assert/data_4�
$assert_shapes/Assert_3/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2&
$assert_shapes/Assert_3/Assert/data_6�
assert_shapes/Assert_3/AssertAssertassert_shapes/Equal_5:z:0-assert_shapes/Assert_3/Assert/data_0:output:0-assert_shapes/Assert_3/Assert/data_1:output:0-assert_shapes/Assert_3/Assert/data_2:output:0-assert_shapes/Assert_3/Assert/data_3:output:0-assert_shapes/Assert_3/Assert/data_4:output:0$assert_shapes/strided_slice:output:0-assert_shapes/Assert_3/Assert/data_6:output:0assert_shapes/Shape_7:output:0^assert_shapes/Assert_2/Assert*
T

2*
_output_shapes
 2
assert_shapes/Assert_3/Assert�

group_depsNoOp^assert_shapes/Assert/Assert^assert_shapes/Assert_1/Assert^assert_shapes/Assert_2/Assert^assert_shapes/Assert_3/Assert:^assert_shapes/assert_rank/static_checks_determined_all_ok<^assert_shapes/assert_rank_1/static_checks_determined_all_ok1^assert_shapes/assert_rank_at_least/Assert/Assert*
_output_shapes
 2

group_depsZ
Shape_11Shapetranspose:y:0*
T0*#
_output_shapes
:���������2

Shape_11z
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_17/stack�
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_17/stack_1~
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_17/stack_2�
strided_slice_17StridedSliceShape_11:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask2
strided_slice_17e
Shape_12Const*
_output_shapes
:*
dtype0*
valueB"2   2   2

Shape_12`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis�
concat_4ConcatV2strided_slice_17:output:0Shape_12:output:0concat_4/axis:output:0*
N*
T0*#
_output_shapes
:���������2

concat_4r
BroadcastToBroadcastToCholesky:output:0concat_4:output:0*
T0*
_output_shapes
:2
BroadcastTo�
&triangular_solve/MatrixTriangularSolveMatrixTriangularSolveBroadcastTo:output:0transpose:y:0*
T0*
_output_shapes
:2(
&triangular_solve/MatrixTriangularSolver
Square_3Square/triangular_solve/MatrixTriangularSolve:output:0*
T0*
_output_shapes
:2

Square_3}
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum_3/reduction_indicesh
Sum_3SumSquare_3:y:0 Sum_3/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_3V
sub_4SubAddN_1:sum:0Sum_3:output:0*
T0*
_output_shapes
:2
sub_4�
concat_5/values_1Packstrided_slice_14:output:0strided_slice_15:output:0*
N*
T0*
_output_shapes
:2
concat_5/values_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis�
concat_5ConcatV2strided_slice_17:output:0concat_5/values_1:output:0concat_5/axis:output:0*
N*
T0*#
_output_shapes
:���������2

concat_5k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
ExpandDims/dimm

ExpandDims
ExpandDims	sub_4:z:0ExpandDims/dim:output:0*
T0*
_output_shapes
:2

ExpandDimsx
BroadcastTo_1BroadcastToExpandDims:output:0concat_5:output:0*
T0*
_output_shapes
:2
BroadcastTo_1�
adjoint_1/matrix_transpose/RankRankBroadcastTo:output:0*
T0*
_output_shapes
: 2!
adjoint_1/matrix_transpose/Rank�
 adjoint_1/matrix_transpose/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 adjoint_1/matrix_transpose/sub/y�
adjoint_1/matrix_transpose/subSub(adjoint_1/matrix_transpose/Rank:output:0)adjoint_1/matrix_transpose/sub/y:output:0*
T0*
_output_shapes
: 2 
adjoint_1/matrix_transpose/sub�
&adjoint_1/matrix_transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 2(
&adjoint_1/matrix_transpose/Range/start�
&adjoint_1/matrix_transpose/Range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2(
&adjoint_1/matrix_transpose/Range/delta�
 adjoint_1/matrix_transpose/RangeRange/adjoint_1/matrix_transpose/Range/start:output:0"adjoint_1/matrix_transpose/sub:z:0/adjoint_1/matrix_transpose/Range/delta:output:0*#
_output_shapes
:���������2"
 adjoint_1/matrix_transpose/Range�
"adjoint_1/matrix_transpose/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"adjoint_1/matrix_transpose/sub_1/y�
 adjoint_1/matrix_transpose/sub_1Sub(adjoint_1/matrix_transpose/Rank:output:0+adjoint_1/matrix_transpose/sub_1/y:output:0*
T0*
_output_shapes
: 2"
 adjoint_1/matrix_transpose/sub_1�
"adjoint_1/matrix_transpose/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"adjoint_1/matrix_transpose/sub_2/y�
 adjoint_1/matrix_transpose/sub_2Sub(adjoint_1/matrix_transpose/Rank:output:0+adjoint_1/matrix_transpose/sub_2/y:output:0*
T0*
_output_shapes
: 2"
 adjoint_1/matrix_transpose/sub_2�
*adjoint_1/matrix_transpose/concat/values_1Pack$adjoint_1/matrix_transpose/sub_1:z:0$adjoint_1/matrix_transpose/sub_2:z:0*
N*
T0*
_output_shapes
:2,
*adjoint_1/matrix_transpose/concat/values_1�
&adjoint_1/matrix_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&adjoint_1/matrix_transpose/concat/axis�
!adjoint_1/matrix_transpose/concatConcatV2)adjoint_1/matrix_transpose/Range:output:03adjoint_1/matrix_transpose/concat/values_1:output:0/adjoint_1/matrix_transpose/concat/axis:output:0*
N*
T0*#
_output_shapes
:���������2#
!adjoint_1/matrix_transpose/concat�
$adjoint_1/matrix_transpose/transpose	TransposeBroadcastTo:output:0*adjoint_1/matrix_transpose/concat:output:0*
T0*
_output_shapes
:2&
$adjoint_1/matrix_transpose/transpose�
(triangular_solve_1/MatrixTriangularSolveMatrixTriangularSolve(adjoint_1/matrix_transpose/transpose:y:0/triangular_solve/MatrixTriangularSolve:output:0*
T0*
_output_shapes
:*
lower( 2*
(triangular_solve_1/MatrixTriangularSolve�
concat_6/values_1Packstrided_slice_16:output:0strided_slice_14:output:0*
N*
T0*
_output_shapes
:2
concat_6/values_1`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_6/axis�
concat_6ConcatV2strided_slice_17:output:0concat_6/values_1:output:0concat_6/axis:output:0*
N*
T0*#
_output_shapes
:���������2

concat_6l
BroadcastTo_2BroadcastTosub:z:0concat_6:output:0*
T0*
_output_shapes
:2
BroadcastTo_2�
MatMul_1BatchMatMulV21triangular_solve_1/MatrixTriangularSolve:output:0BroadcastTo_2:output:0*
T0*
_output_shapes
:*
adj_x(2

MatMul_1�
adjoint_2/matrix_transpose/RankRankBroadcastTo_1:output:0*
T0*
_output_shapes
: 2!
adjoint_2/matrix_transpose/Rank�
 adjoint_2/matrix_transpose/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 adjoint_2/matrix_transpose/sub/y�
adjoint_2/matrix_transpose/subSub(adjoint_2/matrix_transpose/Rank:output:0)adjoint_2/matrix_transpose/sub/y:output:0*
T0*
_output_shapes
: 2 
adjoint_2/matrix_transpose/sub�
&adjoint_2/matrix_transpose/Range/startConst*
_output_shapes
: *
dtype0*
value	B : 2(
&adjoint_2/matrix_transpose/Range/start�
&adjoint_2/matrix_transpose/Range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2(
&adjoint_2/matrix_transpose/Range/delta�
 adjoint_2/matrix_transpose/RangeRange/adjoint_2/matrix_transpose/Range/start:output:0"adjoint_2/matrix_transpose/sub:z:0/adjoint_2/matrix_transpose/Range/delta:output:0*#
_output_shapes
:���������2"
 adjoint_2/matrix_transpose/Range�
"adjoint_2/matrix_transpose/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"adjoint_2/matrix_transpose/sub_1/y�
 adjoint_2/matrix_transpose/sub_1Sub(adjoint_2/matrix_transpose/Rank:output:0+adjoint_2/matrix_transpose/sub_1/y:output:0*
T0*
_output_shapes
: 2"
 adjoint_2/matrix_transpose/sub_1�
"adjoint_2/matrix_transpose/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"adjoint_2/matrix_transpose/sub_2/y�
 adjoint_2/matrix_transpose/sub_2Sub(adjoint_2/matrix_transpose/Rank:output:0+adjoint_2/matrix_transpose/sub_2/y:output:0*
T0*
_output_shapes
: 2"
 adjoint_2/matrix_transpose/sub_2�
*adjoint_2/matrix_transpose/concat/values_1Pack$adjoint_2/matrix_transpose/sub_1:z:0$adjoint_2/matrix_transpose/sub_2:z:0*
N*
T0*
_output_shapes
:2,
*adjoint_2/matrix_transpose/concat/values_1�
&adjoint_2/matrix_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&adjoint_2/matrix_transpose/concat/axis�
!adjoint_2/matrix_transpose/concatConcatV2)adjoint_2/matrix_transpose/Range:output:03adjoint_2/matrix_transpose/concat/values_1:output:0/adjoint_2/matrix_transpose/concat/axis:output:0*
N*
T0*#
_output_shapes
:���������2#
!adjoint_2/matrix_transpose/concat�
$adjoint_2/matrix_transpose/transpose	TransposeBroadcastTo_1:output:0*adjoint_2/matrix_transpose/concat:output:0*
T0*
_output_shapes
:2&
$adjoint_2/matrix_transpose/transposet
assert_shapes_1/ShapeShapetranspose:y:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shaped
assert_shapes_1/RankRanktranspose:y:0*
T0*
_output_shapes
: 2
assert_shapes_1/Rankt
assert_shapes_1/Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_shapes_1/Equal/y�
assert_shapes_1/EqualEqualassert_shapes_1/Rank:output:0 assert_shapes_1/Equal/y:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal�
assert_shapes_1/condStatelessIfassert_shapes_1/Equal:z:0assert_shapes_1/Shape:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *4
else_branch%R#
!assert_shapes_1_cond_false_116194*"
output_shapes
:���������*3
then_branch$R"
 assert_shapes_1_cond_true_1161932
assert_shapes_1/cond�
assert_shapes_1/cond/IdentityIdentityassert_shapes_1/cond:output:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/cond/Identity�
assert_shapes_1/Shape_1ShapeBroadcastTo_2:output:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_1q
assert_shapes_1/Rank_1RankBroadcastTo_2:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Rank_1x
assert_shapes_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_shapes_1/Equal_1/y�
assert_shapes_1/Equal_1Equalassert_shapes_1/Rank_1:output:0"assert_shapes_1/Equal_1/y:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_1�
assert_shapes_1/cond_1StatelessIfassert_shapes_1/Equal_1:z:0 assert_shapes_1/Shape_1:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
else_branch'R%
#assert_shapes_1_cond_1_false_116209*"
output_shapes
:���������*5
then_branch&R$
"assert_shapes_1_cond_1_true_1162082
assert_shapes_1/cond_1�
assert_shapes_1/cond_1/IdentityIdentityassert_shapes_1/cond_1:output:0*
T0*#
_output_shapes
:���������2!
assert_shapes_1/cond_1/Identity|
assert_shapes_1/Shape_2ShapeMatMul_1:output:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_2l
assert_shapes_1/Rank_2RankMatMul_1:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Rank_2x
assert_shapes_1/Equal_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_shapes_1/Equal_2/y�
assert_shapes_1/Equal_2Equalassert_shapes_1/Rank_2:output:0"assert_shapes_1/Equal_2/y:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_2�
assert_shapes_1/cond_2StatelessIfassert_shapes_1/Equal_2:z:0 assert_shapes_1/Shape_2:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
else_branch'R%
#assert_shapes_1_cond_2_false_116224*"
output_shapes
:���������*5
then_branch&R$
"assert_shapes_1_cond_2_true_1162232
assert_shapes_1/cond_2�
assert_shapes_1/cond_2/IdentityIdentityassert_shapes_1/cond_2:output:0*
T0*#
_output_shapes
:���������2!
assert_shapes_1/cond_2/Identity�
assert_shapes_1/Shape_3Shape(adjoint_2/matrix_transpose/transpose:y:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_3�
assert_shapes_1/Rank_3Rank(adjoint_2/matrix_transpose/transpose:y:0*
T0*
_output_shapes
: 2
assert_shapes_1/Rank_3x
assert_shapes_1/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_shapes_1/Equal_3/y�
assert_shapes_1/Equal_3Equalassert_shapes_1/Rank_3:output:0"assert_shapes_1/Equal_3/y:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_3�
assert_shapes_1/cond_3StatelessIfassert_shapes_1/Equal_3:z:0 assert_shapes_1/Shape_3:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*#
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
else_branch'R%
#assert_shapes_1_cond_3_false_116239*"
output_shapes
:���������*5
then_branch&R$
"assert_shapes_1_cond_3_true_1162382
assert_shapes_1/cond_3�
assert_shapes_1/cond_3/IdentityIdentityassert_shapes_1/cond_3:output:0*
T0*#
_output_shapes
:���������2!
assert_shapes_1/cond_3/Identity�
)assert_shapes_1/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :2+
)assert_shapes_1/assert_rank_at_least/rank�
*assert_shapes_1/assert_rank_at_least/ShapeShapetranspose:y:0*
T0*#
_output_shapes
:���������2,
*assert_shapes_1/assert_rank_at_least/Shape�
Sassert_shapes_1/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2U
Sassert_shapes_1/assert_rank_at_least/assert_type/statically_determined_correct_type�
+assert_shapes_1/assert_rank_at_least/Rank_1Ranktranspose:y:0*
T0*
_output_shapes
: 2-
+assert_shapes_1/assert_rank_at_least/Rank_1�
1assert_shapes_1/assert_rank_at_least/GreaterEqualGreaterEqual4assert_shapes_1/assert_rank_at_least/Rank_1:output:02assert_shapes_1/assert_rank_at_least/rank:output:0*
T0*
_output_shapes
: 23
1assert_shapes_1/assert_rank_at_least/GreaterEqual�
1assert_shapes_1/assert_rank_at_least/Assert/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values23
1assert_shapes_1/assert_rank_at_least/Assert/Const�
3assert_shapes_1/assert_rank_at_least/Assert/Const_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Tensor transpose:0 must have rank at least25
3assert_shapes_1/assert_rank_at_least/Assert/Const_1�
3assert_shapes_1/assert_rank_at_least/Assert/Const_2Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 25
3assert_shapes_1/assert_rank_at_least/Assert/Const_2�
9assert_shapes_1/assert_rank_at_least/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2;
9assert_shapes_1/assert_rank_at_least/Assert/Assert/data_0�
9assert_shapes_1/assert_rank_at_least/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Tensor transpose:0 must have rank at least2;
9assert_shapes_1/assert_rank_at_least/Assert/Assert/data_1�
9assert_shapes_1/assert_rank_at_least/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2;
9assert_shapes_1/assert_rank_at_least/Assert/Assert/data_3�
2assert_shapes_1/assert_rank_at_least/Assert/AssertAssert5assert_shapes_1/assert_rank_at_least/GreaterEqual:z:0Bassert_shapes_1/assert_rank_at_least/Assert/Assert/data_0:output:0Bassert_shapes_1/assert_rank_at_least/Assert/Assert/data_1:output:02assert_shapes_1/assert_rank_at_least/rank:output:0Bassert_shapes_1/assert_rank_at_least/Assert/Assert/data_3:output:03assert_shapes_1/assert_rank_at_least/Shape:output:0^assert_shapes/Assert_3/Assert*
T	
2*
_output_shapes
 24
2assert_shapes_1/assert_rank_at_least/Assert/Assert�
+assert_shapes_1/assert_rank_at_least_1/rankConst*
_output_shapes
: *
dtype0*
value	B :2-
+assert_shapes_1/assert_rank_at_least_1/rank�
,assert_shapes_1/assert_rank_at_least_1/ShapeShapeBroadcastTo_2:output:0*
T0*#
_output_shapes
:���������2.
,assert_shapes_1/assert_rank_at_least_1/Shape�
Uassert_shapes_1/assert_rank_at_least_1/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2W
Uassert_shapes_1/assert_rank_at_least_1/assert_type/statically_determined_correct_type�
-assert_shapes_1/assert_rank_at_least_1/Rank_1RankBroadcastTo_2:output:0*
T0*
_output_shapes
: 2/
-assert_shapes_1/assert_rank_at_least_1/Rank_1�
3assert_shapes_1/assert_rank_at_least_1/GreaterEqualGreaterEqual6assert_shapes_1/assert_rank_at_least_1/Rank_1:output:04assert_shapes_1/assert_rank_at_least_1/rank:output:0*
T0*
_output_shapes
: 25
3assert_shapes_1/assert_rank_at_least_1/GreaterEqual�
3assert_shapes_1/assert_rank_at_least_1/Assert/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values25
3assert_shapes_1/assert_rank_at_least_1/Assert/Const�
5assert_shapes_1/assert_rank_at_least_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*?
value6B4 B.Tensor BroadcastTo_2:0 must have rank at least27
5assert_shapes_1/assert_rank_at_least_1/Assert/Const_1�
5assert_shapes_1/assert_rank_at_least_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 27
5assert_shapes_1/assert_rank_at_least_1/Assert/Const_2�
;assert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2=
;assert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_0�
;assert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*?
value6B4 B.Tensor BroadcastTo_2:0 must have rank at least2=
;assert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_1�
;assert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2=
;assert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_3�
4assert_shapes_1/assert_rank_at_least_1/Assert/AssertAssert7assert_shapes_1/assert_rank_at_least_1/GreaterEqual:z:0Dassert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_0:output:0Dassert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_1:output:04assert_shapes_1/assert_rank_at_least_1/rank:output:0Dassert_shapes_1/assert_rank_at_least_1/Assert/Assert/data_3:output:05assert_shapes_1/assert_rank_at_least_1/Shape:output:03^assert_shapes_1/assert_rank_at_least/Assert/Assert*
T	
2*
_output_shapes
 26
4assert_shapes_1/assert_rank_at_least_1/Assert/Assert�
+assert_shapes_1/assert_rank_at_least_2/rankConst*
_output_shapes
: *
dtype0*
value	B :2-
+assert_shapes_1/assert_rank_at_least_2/rank�
,assert_shapes_1/assert_rank_at_least_2/ShapeShapeMatMul_1:output:0*
T0*#
_output_shapes
:���������2.
,assert_shapes_1/assert_rank_at_least_2/Shape�
Uassert_shapes_1/assert_rank_at_least_2/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2W
Uassert_shapes_1/assert_rank_at_least_2/assert_type/statically_determined_correct_type�
-assert_shapes_1/assert_rank_at_least_2/Rank_1RankMatMul_1:output:0*
T0*
_output_shapes
: 2/
-assert_shapes_1/assert_rank_at_least_2/Rank_1�
3assert_shapes_1/assert_rank_at_least_2/GreaterEqualGreaterEqual6assert_shapes_1/assert_rank_at_least_2/Rank_1:output:04assert_shapes_1/assert_rank_at_least_2/rank:output:0*
T0*
_output_shapes
: 25
3assert_shapes_1/assert_rank_at_least_2/GreaterEqual�
3assert_shapes_1/assert_rank_at_least_2/Assert/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values25
3assert_shapes_1/assert_rank_at_least_2/Assert/Const�
5assert_shapes_1/assert_rank_at_least_2/Assert/Const_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)Tensor MatMul_1:0 must have rank at least27
5assert_shapes_1/assert_rank_at_least_2/Assert/Const_1�
5assert_shapes_1/assert_rank_at_least_2/Assert/Const_2Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 27
5assert_shapes_1/assert_rank_at_least_2/Assert/Const_2�
;assert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2=
;assert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_0�
;assert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)Tensor MatMul_1:0 must have rank at least2=
;assert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_1�
;assert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2=
;assert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_3�
4assert_shapes_1/assert_rank_at_least_2/Assert/AssertAssert7assert_shapes_1/assert_rank_at_least_2/GreaterEqual:z:0Dassert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_0:output:0Dassert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_1:output:04assert_shapes_1/assert_rank_at_least_2/rank:output:0Dassert_shapes_1/assert_rank_at_least_2/Assert/Assert/data_3:output:05assert_shapes_1/assert_rank_at_least_2/Shape:output:05^assert_shapes_1/assert_rank_at_least_1/Assert/Assert*
T	
2*
_output_shapes
 26
4assert_shapes_1/assert_rank_at_least_2/Assert/Assert�
+assert_shapes_1/assert_rank_at_least_3/rankConst*
_output_shapes
: *
dtype0*
value	B :2-
+assert_shapes_1/assert_rank_at_least_3/rank�
,assert_shapes_1/assert_rank_at_least_3/ShapeShape(adjoint_2/matrix_transpose/transpose:y:0*
T0*#
_output_shapes
:���������2.
,assert_shapes_1/assert_rank_at_least_3/Shape�
Uassert_shapes_1/assert_rank_at_least_3/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2W
Uassert_shapes_1/assert_rank_at_least_3/assert_type/statically_determined_correct_type�
-assert_shapes_1/assert_rank_at_least_3/Rank_1Rank(adjoint_2/matrix_transpose/transpose:y:0*
T0*
_output_shapes
: 2/
-assert_shapes_1/assert_rank_at_least_3/Rank_1�
3assert_shapes_1/assert_rank_at_least_3/GreaterEqualGreaterEqual6assert_shapes_1/assert_rank_at_least_3/Rank_1:output:04assert_shapes_1/assert_rank_at_least_3/rank:output:0*
T0*
_output_shapes
: 25
3assert_shapes_1/assert_rank_at_least_3/GreaterEqual�
3assert_shapes_1/assert_rank_at_least_3/Assert/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values25
3assert_shapes_1/assert_rank_at_least_3/Assert/Const�
5assert_shapes_1/assert_rank_at_least_3/Assert/Const_1Const*
_output_shapes
: *
dtype0*V
valueMBK BETensor adjoint_2/matrix_transpose/transpose:0 must have rank at least27
5assert_shapes_1/assert_rank_at_least_3/Assert/Const_1�
5assert_shapes_1/assert_rank_at_least_3/Assert/Const_2Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 27
5assert_shapes_1/assert_rank_at_least_3/Assert/Const_2�
;assert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2=
;assert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_0�
;assert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*V
valueMBK BETensor adjoint_2/matrix_transpose/transpose:0 must have rank at least2=
;assert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_1�
;assert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2=
;assert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_3�
4assert_shapes_1/assert_rank_at_least_3/Assert/AssertAssert7assert_shapes_1/assert_rank_at_least_3/GreaterEqual:z:0Dassert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_0:output:0Dassert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_1:output:04assert_shapes_1/assert_rank_at_least_3/rank:output:0Dassert_shapes_1/assert_rank_at_least_3/Assert/Assert/data_3:output:05assert_shapes_1/assert_rank_at_least_3/Shape:output:05^assert_shapes_1/assert_rank_at_least_2/Assert/Assert*
T	
2*
_output_shapes
 26
4assert_shapes_1/assert_rank_at_least_3/Assert/Assert�
#assert_shapes_1/strided_slice/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2%
#assert_shapes_1/strided_slice/stack�
%assert_shapes_1/strided_slice/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice/stack_1�
%assert_shapes_1/strided_slice/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2'
%assert_shapes_1/strided_slice/stack_2�
assert_shapes_1/strided_sliceStridedSlice&assert_shapes_1/cond/Identity:output:0,assert_shapes_1/strided_slice/stack:output:0.assert_shapes_1/strided_slice/stack_1:output:0.assert_shapes_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
assert_shapes_1/strided_slice�
%assert_shapes_1/strided_slice_1/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice_1/stack�
'assert_shapes_1/strided_slice_1/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2)
'assert_shapes_1/strided_slice_1/stack_1�
'assert_shapes_1/strided_slice_1/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2)
'assert_shapes_1/strided_slice_1/stack_2�
assert_shapes_1/strided_slice_1StridedSlice&assert_shapes_1/cond/Identity:output:0.assert_shapes_1/strided_slice_1/stack:output:00assert_shapes_1/strided_slice_1/stack_1:output:00assert_shapes_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
assert_shapes_1/strided_slice_1�
%assert_shapes_1/strided_slice_2/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice_2/stack�
'assert_shapes_1/strided_slice_2/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2)
'assert_shapes_1/strided_slice_2/stack_1�
'assert_shapes_1/strided_slice_2/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2)
'assert_shapes_1/strided_slice_2/stack_2�
assert_shapes_1/strided_slice_2StridedSlice(assert_shapes_1/cond_1/Identity:output:0.assert_shapes_1/strided_slice_2/stack:output:00assert_shapes_1/strided_slice_2/stack_1:output:00assert_shapes_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
assert_shapes_1/strided_slice_2�
assert_shapes_1/Equal_4Equal(assert_shapes_1/strided_slice_2:output:0&assert_shapes_1/strided_slice:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_4�
assert_shapes_1/Shape_4ShapeBroadcastTo_2:output:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_4�
assert_shapes_1/Assert/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2
assert_shapes_1/Assert/Const�
assert_shapes_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22 
assert_shapes_1/Assert/Const_1�
assert_shapes_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*1
value(B& B Tensor BroadcastTo_2:0 dimension2 
assert_shapes_1/Assert/Const_2�
assert_shapes_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2 
assert_shapes_1/Assert/Const_3�
assert_shapes_1/Assert/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2 
assert_shapes_1/Assert/Const_4�
assert_shapes_1/Assert/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2 
assert_shapes_1/Assert/Const_5�
$assert_shapes_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2&
$assert_shapes_1/Assert/Assert/data_0�
$assert_shapes_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -22&
$assert_shapes_1/Assert/Assert/data_1�
$assert_shapes_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*1
value(B& B Tensor BroadcastTo_2:0 dimension2&
$assert_shapes_1/Assert/Assert/data_2�
$assert_shapes_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2&
$assert_shapes_1/Assert/Assert/data_3�
$assert_shapes_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2&
$assert_shapes_1/Assert/Assert/data_4�
$assert_shapes_1/Assert/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2&
$assert_shapes_1/Assert/Assert/data_6�
assert_shapes_1/Assert/AssertAssertassert_shapes_1/Equal_4:z:0-assert_shapes_1/Assert/Assert/data_0:output:0-assert_shapes_1/Assert/Assert/data_1:output:0-assert_shapes_1/Assert/Assert/data_2:output:0-assert_shapes_1/Assert/Assert/data_3:output:0-assert_shapes_1/Assert/Assert/data_4:output:0&assert_shapes_1/strided_slice:output:0-assert_shapes_1/Assert/Assert/data_6:output:0 assert_shapes_1/Shape_4:output:05^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
T

2*
_output_shapes
 2
assert_shapes_1/Assert/Assert�
%assert_shapes_1/strided_slice_3/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice_3/stack�
'assert_shapes_1/strided_slice_3/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2)
'assert_shapes_1/strided_slice_3/stack_1�
'assert_shapes_1/strided_slice_3/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2)
'assert_shapes_1/strided_slice_3/stack_2�
assert_shapes_1/strided_slice_3StridedSlice(assert_shapes_1/cond_1/Identity:output:0.assert_shapes_1/strided_slice_3/stack:output:00assert_shapes_1/strided_slice_3/stack_1:output:00assert_shapes_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
assert_shapes_1/strided_slice_3�
%assert_shapes_1/strided_slice_4/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice_4/stack�
'assert_shapes_1/strided_slice_4/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2)
'assert_shapes_1/strided_slice_4/stack_1�
'assert_shapes_1/strided_slice_4/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2)
'assert_shapes_1/strided_slice_4/stack_2�
assert_shapes_1/strided_slice_4StridedSlice(assert_shapes_1/cond_2/Identity:output:0.assert_shapes_1/strided_slice_4/stack:output:00assert_shapes_1/strided_slice_4/stack_1:output:00assert_shapes_1/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
assert_shapes_1/strided_slice_4�
assert_shapes_1/Equal_5Equal(assert_shapes_1/strided_slice_4:output:0(assert_shapes_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_5|
assert_shapes_1/Shape_5ShapeMatMul_1:output:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_5�
assert_shapes_1/Assert_1/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2 
assert_shapes_1/Assert_1/Const�
 assert_shapes_1/Assert_1/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -12"
 assert_shapes_1/Assert_1/Const_1�
 assert_shapes_1/Assert_1/Const_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor MatMul_1:0 dimension2"
 assert_shapes_1/Assert_1/Const_2�
 assert_shapes_1/Assert_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2"
 assert_shapes_1/Assert_1/Const_3�
 assert_shapes_1/Assert_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2"
 assert_shapes_1/Assert_1/Const_4�
 assert_shapes_1/Assert_1/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2"
 assert_shapes_1/Assert_1/Const_5�
&assert_shapes_1/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2(
&assert_shapes_1/Assert_1/Assert/data_0�
&assert_shapes_1/Assert_1/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -12(
&assert_shapes_1/Assert_1/Assert/data_1�
&assert_shapes_1/Assert_1/Assert/data_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor MatMul_1:0 dimension2(
&assert_shapes_1/Assert_1/Assert/data_2�
&assert_shapes_1/Assert_1/Assert/data_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2(
&assert_shapes_1/Assert_1/Assert/data_3�
&assert_shapes_1/Assert_1/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2(
&assert_shapes_1/Assert_1/Assert/data_4�
&assert_shapes_1/Assert_1/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2(
&assert_shapes_1/Assert_1/Assert/data_6�
assert_shapes_1/Assert_1/AssertAssertassert_shapes_1/Equal_5:z:0/assert_shapes_1/Assert_1/Assert/data_0:output:0/assert_shapes_1/Assert_1/Assert/data_1:output:0/assert_shapes_1/Assert_1/Assert/data_2:output:0/assert_shapes_1/Assert_1/Assert/data_3:output:0/assert_shapes_1/Assert_1/Assert/data_4:output:0(assert_shapes_1/strided_slice_1:output:0/assert_shapes_1/Assert_1/Assert/data_6:output:0 assert_shapes_1/Shape_5:output:0^assert_shapes_1/Assert/Assert*
T

2*
_output_shapes
 2!
assert_shapes_1/Assert_1/Assert�
%assert_shapes_1/strided_slice_5/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice_5/stack�
'assert_shapes_1/strided_slice_5/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2)
'assert_shapes_1/strided_slice_5/stack_1�
'assert_shapes_1/strided_slice_5/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2)
'assert_shapes_1/strided_slice_5/stack_2�
assert_shapes_1/strided_slice_5StridedSlice(assert_shapes_1/cond_2/Identity:output:0.assert_shapes_1/strided_slice_5/stack:output:00assert_shapes_1/strided_slice_5/stack_1:output:00assert_shapes_1/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
assert_shapes_1/strided_slice_5�
assert_shapes_1/Equal_6Equal(assert_shapes_1/strided_slice_5:output:0(assert_shapes_1/strided_slice_3:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_6|
assert_shapes_1/Shape_6ShapeMatMul_1:output:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_6�
assert_shapes_1/Assert_2/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2 
assert_shapes_1/Assert_2/Const�
 assert_shapes_1/Assert_2/Const_1Const*
_output_shapes
: *
dtype0*A
value8B6 B0Specified by tensor BroadcastTo_2:0 dimension -12"
 assert_shapes_1/Assert_2/Const_1�
 assert_shapes_1/Assert_2/Const_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor MatMul_1:0 dimension2"
 assert_shapes_1/Assert_2/Const_2�
 assert_shapes_1/Assert_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2"
 assert_shapes_1/Assert_2/Const_3�
 assert_shapes_1/Assert_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2"
 assert_shapes_1/Assert_2/Const_4�
 assert_shapes_1/Assert_2/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2"
 assert_shapes_1/Assert_2/Const_5�
&assert_shapes_1/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2(
&assert_shapes_1/Assert_2/Assert/data_0�
&assert_shapes_1/Assert_2/Assert/data_1Const*
_output_shapes
: *
dtype0*A
value8B6 B0Specified by tensor BroadcastTo_2:0 dimension -12(
&assert_shapes_1/Assert_2/Assert/data_1�
&assert_shapes_1/Assert_2/Assert/data_2Const*
_output_shapes
: *
dtype0*,
value#B! BTensor MatMul_1:0 dimension2(
&assert_shapes_1/Assert_2/Assert/data_2�
&assert_shapes_1/Assert_2/Assert/data_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2(
&assert_shapes_1/Assert_2/Assert/data_3�
&assert_shapes_1/Assert_2/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2(
&assert_shapes_1/Assert_2/Assert/data_4�
&assert_shapes_1/Assert_2/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2(
&assert_shapes_1/Assert_2/Assert/data_6�
assert_shapes_1/Assert_2/AssertAssertassert_shapes_1/Equal_6:z:0/assert_shapes_1/Assert_2/Assert/data_0:output:0/assert_shapes_1/Assert_2/Assert/data_1:output:0/assert_shapes_1/Assert_2/Assert/data_2:output:0/assert_shapes_1/Assert_2/Assert/data_3:output:0/assert_shapes_1/Assert_2/Assert/data_4:output:0(assert_shapes_1/strided_slice_3:output:0/assert_shapes_1/Assert_2/Assert/data_6:output:0 assert_shapes_1/Shape_6:output:0 ^assert_shapes_1/Assert_1/Assert*
T

2*
_output_shapes
 2!
assert_shapes_1/Assert_2/Assert�
%assert_shapes_1/strided_slice_6/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice_6/stack�
'assert_shapes_1/strided_slice_6/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2)
'assert_shapes_1/strided_slice_6/stack_1�
'assert_shapes_1/strided_slice_6/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2)
'assert_shapes_1/strided_slice_6/stack_2�
assert_shapes_1/strided_slice_6StridedSlice(assert_shapes_1/cond_3/Identity:output:0.assert_shapes_1/strided_slice_6/stack:output:00assert_shapes_1/strided_slice_6/stack_1:output:00assert_shapes_1/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
assert_shapes_1/strided_slice_6�
assert_shapes_1/Equal_7Equal(assert_shapes_1/strided_slice_6:output:0(assert_shapes_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_7�
assert_shapes_1/Shape_7Shape(adjoint_2/matrix_transpose/transpose:y:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_7�
assert_shapes_1/Assert_3/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2 
assert_shapes_1/Assert_3/Const�
 assert_shapes_1/Assert_3/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -12"
 assert_shapes_1/Assert_3/Const_1�
 assert_shapes_1/Assert_3/Const_2Const*
_output_shapes
: *
dtype0*H
value?B= B7Tensor adjoint_2/matrix_transpose/transpose:0 dimension2"
 assert_shapes_1/Assert_3/Const_2�
 assert_shapes_1/Assert_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2"
 assert_shapes_1/Assert_3/Const_3�
 assert_shapes_1/Assert_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2"
 assert_shapes_1/Assert_3/Const_4�
 assert_shapes_1/Assert_3/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2"
 assert_shapes_1/Assert_3/Const_5�
&assert_shapes_1/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2(
&assert_shapes_1/Assert_3/Assert/data_0�
&assert_shapes_1/Assert_3/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,Specified by tensor transpose:0 dimension -12(
&assert_shapes_1/Assert_3/Assert/data_1�
&assert_shapes_1/Assert_3/Assert/data_2Const*
_output_shapes
: *
dtype0*H
value?B= B7Tensor adjoint_2/matrix_transpose/transpose:0 dimension2(
&assert_shapes_1/Assert_3/Assert/data_2�
&assert_shapes_1/Assert_3/Assert/data_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2(
&assert_shapes_1/Assert_3/Assert/data_3�
&assert_shapes_1/Assert_3/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2(
&assert_shapes_1/Assert_3/Assert/data_4�
&assert_shapes_1/Assert_3/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2(
&assert_shapes_1/Assert_3/Assert/data_6�
assert_shapes_1/Assert_3/AssertAssertassert_shapes_1/Equal_7:z:0/assert_shapes_1/Assert_3/Assert/data_0:output:0/assert_shapes_1/Assert_3/Assert/data_1:output:0/assert_shapes_1/Assert_3/Assert/data_2:output:0/assert_shapes_1/Assert_3/Assert/data_3:output:0/assert_shapes_1/Assert_3/Assert/data_4:output:0(assert_shapes_1/strided_slice_1:output:0/assert_shapes_1/Assert_3/Assert/data_6:output:0 assert_shapes_1/Shape_7:output:0 ^assert_shapes_1/Assert_2/Assert*
T

2*
_output_shapes
 2!
assert_shapes_1/Assert_3/Assert�
%assert_shapes_1/strided_slice_7/stackConst3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:
���������2'
%assert_shapes_1/strided_slice_7/stack�
'assert_shapes_1/strided_slice_7/stack_1Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2)
'assert_shapes_1/strided_slice_7/stack_1�
'assert_shapes_1/strided_slice_7/stack_2Const3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB:2)
'assert_shapes_1/strided_slice_7/stack_2�
assert_shapes_1/strided_slice_7StridedSlice(assert_shapes_1/cond_3/Identity:output:0.assert_shapes_1/strided_slice_7/stack:output:00assert_shapes_1/strided_slice_7/stack_1:output:00assert_shapes_1/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
assert_shapes_1/strided_slice_7�
assert_shapes_1/Equal_8Equal(assert_shapes_1/strided_slice_7:output:0(assert_shapes_1/strided_slice_3:output:0*
T0*
_output_shapes
: 2
assert_shapes_1/Equal_8�
assert_shapes_1/Shape_8Shape(adjoint_2/matrix_transpose/transpose:y:0*
T0*#
_output_shapes
:���������2
assert_shapes_1/Shape_8�
assert_shapes_1/Assert_4/ConstConst*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2 
assert_shapes_1/Assert_4/Const�
 assert_shapes_1/Assert_4/Const_1Const*
_output_shapes
: *
dtype0*A
value8B6 B0Specified by tensor BroadcastTo_2:0 dimension -12"
 assert_shapes_1/Assert_4/Const_1�
 assert_shapes_1/Assert_4/Const_2Const*
_output_shapes
: *
dtype0*H
value?B= B7Tensor adjoint_2/matrix_transpose/transpose:0 dimension2"
 assert_shapes_1/Assert_4/Const_2�
 assert_shapes_1/Assert_4/Const_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2"
 assert_shapes_1/Assert_4/Const_3�
 assert_shapes_1/Assert_4/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2"
 assert_shapes_1/Assert_4/Const_4�
 assert_shapes_1/Assert_4/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2"
 assert_shapes_1/Assert_4/Const_5�
&assert_shapes_1/Assert_4/Assert/data_0Const*
_output_shapes
: *
dtype0*1
value(B& B base_conditional() return values2(
&assert_shapes_1/Assert_4/Assert/data_0�
&assert_shapes_1/Assert_4/Assert/data_1Const*
_output_shapes
: *
dtype0*A
value8B6 B0Specified by tensor BroadcastTo_2:0 dimension -12(
&assert_shapes_1/Assert_4/Assert/data_1�
&assert_shapes_1/Assert_4/Assert/data_2Const*
_output_shapes
: *
dtype0*H
value?B= B7Tensor adjoint_2/matrix_transpose/transpose:0 dimension2(
&assert_shapes_1/Assert_4/Assert/data_2�
&assert_shapes_1/Assert_4/Assert/data_3Const*
_output_shapes
: *
dtype0*
valueB :
���������2(
&assert_shapes_1/Assert_4/Assert/data_3�
&assert_shapes_1/Assert_4/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size2(
&assert_shapes_1/Assert_4/Assert/data_4�
&assert_shapes_1/Assert_4/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: 2(
&assert_shapes_1/Assert_4/Assert/data_6�
assert_shapes_1/Assert_4/AssertAssertassert_shapes_1/Equal_8:z:0/assert_shapes_1/Assert_4/Assert/data_0:output:0/assert_shapes_1/Assert_4/Assert/data_1:output:0/assert_shapes_1/Assert_4/Assert/data_2:output:0/assert_shapes_1/Assert_4/Assert/data_3:output:0/assert_shapes_1/Assert_4/Assert/data_4:output:0(assert_shapes_1/strided_slice_3:output:0/assert_shapes_1/Assert_4/Assert/data_6:output:0 assert_shapes_1/Shape_8:output:0 ^assert_shapes_1/Assert_3/Assert*
T

2*
_output_shapes
 2!
assert_shapes_1/Assert_4/Assert�
group_deps_1NoOp^assert_shapes_1/Assert/Assert ^assert_shapes_1/Assert_1/Assert ^assert_shapes_1/Assert_2/Assert ^assert_shapes_1/Assert_3/Assert ^assert_shapes_1/Assert_4/Assert3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert*
_output_shapes
 2
group_deps_1Q
Shape_13Shapexnew*
T0*#
_output_shapes
:���������2

Shape_13z
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_18/stack�
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_18/stack_1~
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack_2�
strided_slice_18StridedSliceShape_13:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask2
strided_slice_18p
concat_7/values_1Const*
_output_shapes
:*
dtype0*
valueB:2
concat_7/values_1`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_7/axis�
concat_7ConcatV2strided_slice_18:output:0concat_7/values_1:output:0concat_7/axis:output:0*
N*
T0*#
_output_shapes
:���������2

concat_7g
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB 2        2
zeros_2/Consth
zeros_2Fillconcat_7:output:0zeros_2/Const:output:0*
T0*
_output_shapes
:2	
zeros_2_
add_5AddV2MatMul_1:output:0zeros_2:output:0*
T0*
_output_shapes
:2
add_5�
IdentityIdentity	add_5:z:09^Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp(^Squeeze/softplus/forward/ReadVariableOp*^Squeeze_1/softplus/forward/ReadVariableOp*^Squeeze_2/softplus/forward/ReadVariableOp^assert_shapes/Assert/Assert^assert_shapes/Assert_1/Assert^assert_shapes/Assert_2/Assert^assert_shapes/Assert_3/Assert1^assert_shapes/assert_rank_at_least/Assert/Assert^assert_shapes_1/Assert/Assert ^assert_shapes_1/Assert_1/Assert ^assert_shapes_1/Assert_2/Assert ^assert_shapes_1/Assert_3/Assert ^assert_shapes_1/Assert_4/Assert3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert ^softplus/forward/ReadVariableOp"^softplus_1/forward/ReadVariableOp(^truediv/softplus/forward/ReadVariableOp*^truediv_1/softplus/forward/ReadVariableOp*^truediv_2/softplus/forward/ReadVariableOp*
T0*
_output_shapes
:2

Identity�

Identity_1Identity(adjoint_2/matrix_transpose/transpose:y:09^Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp(^Squeeze/softplus/forward/ReadVariableOp*^Squeeze_1/softplus/forward/ReadVariableOp*^Squeeze_2/softplus/forward/ReadVariableOp^assert_shapes/Assert/Assert^assert_shapes/Assert_1/Assert^assert_shapes/Assert_2/Assert^assert_shapes/Assert_3/Assert1^assert_shapes/assert_rank_at_least/Assert/Assert^assert_shapes_1/Assert/Assert ^assert_shapes_1/Assert_1/Assert ^assert_shapes_1/Assert_2/Assert ^assert_shapes_1/Assert_3/Assert ^assert_shapes_1/Assert_4/Assert3^assert_shapes_1/assert_rank_at_least/Assert/Assert5^assert_shapes_1/assert_rank_at_least_1/Assert/Assert5^assert_shapes_1/assert_rank_at_least_2/Assert/Assert5^assert_shapes_1/assert_rank_at_least_3/Assert/Assert ^softplus/forward/ReadVariableOp"^softplus_1/forward/ReadVariableOp(^truediv/softplus/forward/ReadVariableOp*^truediv_1/softplus/forward/ReadVariableOp*^truediv_2/softplus/forward/ReadVariableOp*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*=
_input_shapes,
*::2:2::::: 2t
8Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp8Fill_3/chain_of_shift_of_softplus/forward/ReadVariableOp2R
'Squeeze/softplus/forward/ReadVariableOp'Squeeze/softplus/forward/ReadVariableOp2V
)Squeeze_1/softplus/forward/ReadVariableOp)Squeeze_1/softplus/forward/ReadVariableOp2V
)Squeeze_2/softplus/forward/ReadVariableOp)Squeeze_2/softplus/forward/ReadVariableOp2:
assert_shapes/Assert/Assertassert_shapes/Assert/Assert2>
assert_shapes/Assert_1/Assertassert_shapes/Assert_1/Assert2>
assert_shapes/Assert_2/Assertassert_shapes/Assert_2/Assert2>
assert_shapes/Assert_3/Assertassert_shapes/Assert_3/Assert2d
0assert_shapes/assert_rank_at_least/Assert/Assert0assert_shapes/assert_rank_at_least/Assert/Assert2>
assert_shapes_1/Assert/Assertassert_shapes_1/Assert/Assert2B
assert_shapes_1/Assert_1/Assertassert_shapes_1/Assert_1/Assert2B
assert_shapes_1/Assert_2/Assertassert_shapes_1/Assert_2/Assert2B
assert_shapes_1/Assert_3/Assertassert_shapes_1/Assert_3/Assert2B
assert_shapes_1/Assert_4/Assertassert_shapes_1/Assert_4/Assert2h
2assert_shapes_1/assert_rank_at_least/Assert/Assert2assert_shapes_1/assert_rank_at_least/Assert/Assert2l
4assert_shapes_1/assert_rank_at_least_1/Assert/Assert4assert_shapes_1/assert_rank_at_least_1/Assert/Assert2l
4assert_shapes_1/assert_rank_at_least_2/Assert/Assert4assert_shapes_1/assert_rank_at_least_2/Assert/Assert2l
4assert_shapes_1/assert_rank_at_least_3/Assert/Assert4assert_shapes_1/assert_rank_at_least_3/Assert/Assert2B
softplus/forward/ReadVariableOpsoftplus/forward/ReadVariableOp2F
!softplus_1/forward/ReadVariableOp!softplus_1/forward/ReadVariableOp2R
'truediv/softplus/forward/ReadVariableOp'truediv/softplus/forward/ReadVariableOp2V
)truediv_1/softplus/forward/ReadVariableOp)truediv_1/softplus/forward/ReadVariableOp2V
)truediv_2/softplus/forward/ReadVariableOp)truediv_2/softplus/forward/ReadVariableOp:> :

_output_shapes
:

_user_specified_nameXnew:$ 

_output_shapes

:2:$ 

_output_shapes

:2:

_output_shapes
: 
�	
�
$__inference_signature_wrapper_116427
xnew
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxnewunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *%
f R
__inference_predict_f_1164042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*=
_input_shapes,
*::2:2::::: 22
StatefulPartitionedCallStatefulPartitionedCall:> :

_output_shapes
:

_user_specified_nameXnew:$ 

_output_shapes

:2:$ 

_output_shapes

:2:

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
&
Xnew
serving_default_Xnew:0-
output_0!
StatefulPartitionedCall:0-
output_1!
StatefulPartitionedCall:1tensorflow/serving/predict:�
u
mean_function

kernel

likelihood

signatures
predict_f_compiled"
_generic_user_object
"
_generic_user_object
+
kernels"
_generic_user_object
,
variance"
_generic_user_object
,
serving_default"
signature_map
.
0
1"
trackable_list_wrapper
[
	_pretransformed_input

_transform_fn

	_bijector"
_generic_user_object
>
variance
lengthscales"
_generic_user_object
,
variance"
_generic_user_object
: 2Variable
.

_bijectors"
_generic_user_object
[
_pretransformed_input
_transform_fn
	_bijector"
_generic_user_object
[
_pretransformed_input
_transform_fn
	_bijector"
_generic_user_object
[
_pretransformed_input
_transform_fn
	_bijector"
_generic_user_object
.
0
1"
trackable_list_wrapper
: 2Variable
"
_generic_user_object
: 2Variable
"
_generic_user_object
: 2Variable
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
�2�
__inference_predict_f_116404�
���
FullArgSpec:
args2�/
jself
jXnew

jfull_cov
jfull_output_cov
varargs
 
varkw
 
defaults�
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�B�
$__inference_signature_wrapper_116427Xnew"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
	J
Const
J	
Const_1
J	
Const_2j
__inference_predict_f_116404J	�
�
�
Xnew
� "�
�	
0
�	
1�
$__inference_signature_wrapper_116427x	&�#
� 
�

Xnew�
Xnew"E�B

output_0�
output_0

output_1�
output_1