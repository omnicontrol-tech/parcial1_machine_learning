import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from utilities import load_css, st_image
from PIL import Image

###### Page configuration
st.set_page_config(page_title='Streamlit Course Kit', layout='wide', page_icon='📦')

###### Page logo and title
st.title("📦 Streamlit Course Kit")
st.sidebar.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png', use_column_width='always')
st.sidebar.markdown('''
### About
This **Streamlit Course Kit** is a collection of educational resources for helping you get started in using, teaching and creating content about [Streamlit](https://streamlit.io/).
''')

##### App templates
colored_header(
  label='App templates',
  description='Here are templates for building single page and multi-page Streamlit apps.',
  color_name='light-blue-70',
)

