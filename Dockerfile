FROM python:3.7.7-slim-buster as builder
LABEL Description="The container is the method to easily deploy GEMG web application. Author: kry127" Vendor="ITMO JetBrains" Version="1.0"
COPY . /GlowingEyesMemeGenerator
WORKDIR /GlowingEyesMemeGenerator
RUN apt-get update
RUN apt-get install -y python-opencv
RUN apt-get clean && apt-get -y autoremove
RUN python -m pip install --upgrade pip
RUN pip install -r /GlowingEyesMemeGenerator/requirements.txt
# delete photos every 15 minutes:
RUN echo "*/15 * * * * /GlowingEyesMemeGenerator/clear_images_cron.sh" >> /etc/crontab
# append allowed hosts
RUN echo "\nglowingeyesmemes.rybin.org \ngemg.rybin.org \n" >> Eyes3/allowed.hosts
RUN python3 manage.py migrate
RUN chmod +x manage.py
RUN chmod 100 start.sh
ENTRYPOINT ["./start.sh"]

EXPOSE 80