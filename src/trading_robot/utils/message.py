# !pip install pushbullet.py
from IPython.core.magic import Magics, magics_class, line_magic
from pushbullet import Pushbullet
import os


@magics_class
class PushbulletMagic(Magics):
    def __init__(self, shell):
        super(PushbulletMagic, self).__init__(shell)
        self.pb = Pushbullet( os.getenv('PUSHBULLET_API_KEY') ) 
    
    @line_magic
    def pushbullet_notify(self, line):
        """
        %pushbullet_notify MESSAGE
        Send a Pushbullet notification with the provided message.
        """
        message = line.strip()
        title = 'Your Jupyter Message'
        
        push = self.pb.push_note(title, message)
        print("Message sent.")

# Register the magic
get_ipython().register_magics(PushbulletMagic)