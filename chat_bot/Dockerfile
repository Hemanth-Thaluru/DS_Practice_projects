From python:3.8

WORKDIR /chat-bot

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./src ./src

CMD [ "python","./src/chat.py" ]

# How to build this image
    # 1. docker build -t my-chat-bot
    # 2. docker run -it my-chat-bot 

