FROM python:3.7
LABEL mail=e.arnautov@gmail.com
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib/ && ./configure --prefix=/usr && make && make install
RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN mkdir logs & mkdir data & mkdir s3
ADD py /app/py
WORKDIR /app/py

CMD [ "python", "start_optimization.py" ]
