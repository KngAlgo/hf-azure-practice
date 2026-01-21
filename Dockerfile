FROM python:3.8-slim 
# find python

WORKDIR /app 
# create a working directory in our image

COPY requirements.txt . 
# copy the requirements
RUN pip install --no-cache-dir -r requirements.txt 
# install the requirements

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]