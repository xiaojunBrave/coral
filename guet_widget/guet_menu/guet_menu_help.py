from guet_handle.guet_help_handle import GuetHelpHandle
from guet_widget.guet_menu.guet_menu import GuetMenu


class GuetMenuHelp(GuetMenu):
    def __init__(self, parent):
        super(GuetMenuHelp, self).__init__(parent)
        self.bind_handle(GuetHelpHandle)