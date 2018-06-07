FROM eduardoslopes/deep_learning

LABEL authors="Eduardo Lopes <eduardo.lopes.es@gmail.com>"

RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

COPY requirements.txt /notebooks
RUN pip install --no-cache-dir -r requirements.txt

RUN jupyter nbextension enable --py --sys-prefix gmaps
RUN jt -t monokai -m 200