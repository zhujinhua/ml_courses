FROM silverlogic/python3.8 as builder
USER root

RUN mkdir /ml_courses
COPY . /ml_courses
WORKDIR /ml_courses

RUN python3 -m venv ~/ml-courses-venv && . ~/ml-courses-venv/bin/activate

RUN pwd & ls -la
# install all python requirements.
RUN python3 -m pip install --default-timeout=100 --retries=10 -r ./requirements.txt