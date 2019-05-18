from tensorflow.contrib import slim
from frontends import resnet_v2
from frontends import mobilenet_v2
from frontends import inception_v4
from frontends import densenet
from frontends import xception
import os 
import subprocess


def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


def build_frontend(inputs, frontend_config, is_training=True, reuse=False):
    frontend = frontend_config['frontend']
    pretrained_dir = frontend_config['pretrained_dir']

    if "ResNet50" == frontend and not os.path.isfile("pretrain/resnet_v2_50.ckpt"):
        download_checkpoints("ResNet50")
    if "ResNet101" == frontend and not os.path.isfile("pretrain/resnet_v2_101.ckpt"):
        download_checkpoints("ResNet101")
    if "ResNet152" == frontend and not os.path.isfile("pretrain/resnet_v2_152.ckpt"):
        download_checkpoints("ResNet152")
    if "MobileNetV2" == frontend and not os.path.isfile("pretrain/mobilenet_v2.ckpt.data-00000-of-00001"):
        download_checkpoints("MobileNetV2")
    if "InceptionV4" == frontend and not os.path.isfile("pretrain/inception_v4.ckpt"):
        download_checkpoints("InceptionV4")

    if frontend == 'ResNet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50', reuse=reuse)
            frontend_scope='resnet_v2_50'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), var_list=slim.get_model_variables('resnet_v2_50'), ignore_missing_vars=True)
    elif frontend == 'ResNet101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101', reuse=reuse)
            frontend_scope='resnet_v2_101'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), var_list=slim.get_model_variables('resnet_v2_101'), ignore_missing_vars=True)
    elif frontend == 'ResNet152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152', reuse=reuse)
            frontend_scope='resnet_v2_152'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), var_list=slim.get_model_variables('resnet_v2_152'), ignore_missing_vars=True)
    elif frontend == 'MobileNetV2':
        with slim.arg_scope(mobilenet_v2.training_scope()):
            logits, end_points = mobilenet_v2.mobilenet(inputs, is_training=is_training, scope='mobilenet_v2', base_only=True, reuse=reuse)
            frontend_scope='mobilenet_v2'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'mobilenet_v2.ckpt'), var_list=slim.get_model_variables('mobilenet_v2'), ignore_missing_vars=True)
    elif frontend == 'InceptionV4':
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = inception_v4.inception_v4(inputs, is_training=is_training, scope='inception_v4', reuse=reuse)
            frontend_scope='inception_v4'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'inception_v4.ckpt'), var_list=slim.get_model_variables('inception_v4'), ignore_missing_vars=True)
    elif frontend == 'DenseNet121':
        with slim.arg_scope(densenet.densenet_arg_scope()):
            logits, end_points = densenet.densenet121(inputs, is_training=is_training, scope='densenet121', reuse=reuse)
            frontend_scope ='densenet121'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'tf-densenet121/tf-densenet121.ckpt'), var_list=slim.get_model_variables('densenet121'), ignore_missing_vars=True)
    elif frontend == 'DenseNet161':
        with slim.arg_scope(densenet.densenet_arg_scope()):
            logits, end_points = densenet.densenet121(inputs, is_training=is_training, scope='densenet161', reuse=reuse)
            frontend_scope='densenet161'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'tf-densenet161.ckpt'), var_list=slim.get_model_variables('densenet161'), ignore_missing_vars=True)
    elif frontend == 'DenseNet169':
        with slim.arg_scope(densenet.densenet_arg_scope()):
            logits, end_points= densenet.densenet121(inputs, is_training=is_training, scope='densenet169', reuse=reuse)
            frontend_scope='densenet169'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'tf-densenet169.ckpt'), var_list=slim.get_model_variables('densenet169'), ignore_missing_vars=True)
    elif frontend == 'Xception39':
        with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception39(inputs, is_training=is_training, scope='xception39', reuse=reuse)
            frontend_scope='Xception39'
            init_fn = None
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152, and MobileNetV2" % (frontend))

    return logits, end_points, frontend_scope, init_fn 