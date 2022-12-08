from guet_handle.guet_toorbar_handle import GuetToorBarHandle
from guet_widget.guet_menu.guet_menu import GuetMenu


class GuetMenuToorBar(GuetMenu):
    def __init__(self, parent):
        super(GuetMenuToorBar, self).__init__(parent)
        self.bind_handle(GuetToorBarHandle)