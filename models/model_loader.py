import streamlit as st
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


@st.cache_resource
def load_paddle_model(model_path):
    """Load the pretrained PaddleOCR model for detection"""
    try:
        model = PaddleOCR(
            det_model_dir=model_path,
            use_angle_cls=True,
            lang='vi',
            rec=False,
            use_gpu=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading PaddleOCR model: {e}")
        return None


@st.cache_resource
def load_vietocr_model():
    """Load the VietOCR model for text recognition"""
    try:
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['device'] = 'cpu'
        # config['cnn']['pretrained'] = True
        config['weights'] = 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
        config['pretrain'] = 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
        predictor = Predictor(config)
        return predictor
    except Exception as e:
        st.error(f"Error loading VietOCR model: {e}")
        return None
