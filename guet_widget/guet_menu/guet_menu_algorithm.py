from guet_widget.guet_menu.guet_menu import GuetMenu
from  guet_handle.guet_algorithm_handle import GuetAlgorithmHandle


class GuetMenuAlgorithm(GuetMenu):
    def __init__(self, parent):
        super(GuetMenuAlgorithm, self).__init__(parent)
        self.bind_handle(GuetAlgorithmHandle)
