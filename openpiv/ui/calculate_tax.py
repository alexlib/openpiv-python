#!/usr/bin/env python
import sys
# from PyQt import QtCore, QtGui, uic
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
 
qtCreatorFile = "tax.ui" # Enter file here.
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

 
class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.calc_tax_button.clicked.connect(self.CalculateTax)
    
    def CalculateTax(self):
        price = int(self.ui.price_box.toPlainText())
        tax = (self.ui.tax_rate.value())
        total_price = price  + ((tax / 100.) * price)
        total_price_string = "The total price with tax is: " + str(total_price)
        print(total_price_string)
        self.ui.results_window.setText(total_price_string)
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())