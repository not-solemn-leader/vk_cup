FROM continuumio/anaconda3:latest

CMD mkdir /opt/results
RUN conda config --add channels conda-forge
RUN conda install catboost

WORKDIR /opt/results

COPY v13.py v13.py
COPY data data
COPY tuned_ensemble tuned_ensemble

CMD python v13.py /tmp/data/test.tsv > /opt/results/result.tsv