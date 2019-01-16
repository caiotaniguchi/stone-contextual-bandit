FROM jupyter/scipy-notebook

RUN conda install --quiet --yes \
    'scikit-learn=0.20*' \
    'plotly=3.*' \
    'xgboost=0.80*' && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    pip install contextualbandits && \
    jupyter lab build
