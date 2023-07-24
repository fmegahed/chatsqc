css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin_bottom: 1rem; display: flex;
}

.chat-message.user {
    background-color: #dedcdc;
    margin_bottom: 10rem;
}

.chat-message.bot {
    border-style: solid; border-color: #C3142D;
    background-color: #FFFFFF;
    margin_bottom: 3rem;
}

.chat-message .avatar {
    width: 15%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}


.chat-message .message{
    width: 85%;
    padding: 0 1.5rem;
    color: #000000;
}
'''

bot_template = '''
<div class='chat-message bot'>
    <div class ="avatar">
        <img src="https://github.com/fmegahed/ai_imgs/blob/main/chatbot.png?raw=true">
    </div>
    <div class = "message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class='chat-message user'>
    <div class ="avatar">
        <img src="https://github.com/fmegahed/ai_imgs/blob/main/user.png?raw=true">
    </div>
    <div class = "message">{{MSG}}</div>
</div>
'''
