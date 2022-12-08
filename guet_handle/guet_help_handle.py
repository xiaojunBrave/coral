from guet_handle.guet_handle import GuetHandle
from py_files.HelpPaper import HelpPaperWindow


class GuetHelpHandle(GuetHandle):
    def called(self, action):
        super().called(action)

        if action.text() == "帮助文档":
            self.open_helppaper()

    def open_helppaper(self):
        self.helppaper = HelpPaperWindow()
        self.helppaper.show()
