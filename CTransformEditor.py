import sys
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QColorDialog, \
    QPushButton, QLabel, QCheckBox, QComboBox, QSlider, QFormLayout, QTabWidget, QSpinBox
from PyQt5.QtGui import QPainter, QColor, QFont, QPixmap, QImage
from PyQt5.QtCore import Qt

from CTransformGrid import Range, Grid, TFunctions, TMatrix

import numpy as np



class CellRC:
    def __init__(self, r, c):
        self.r = r
        self.c = c


class ActiveCell:
    def __init__(self, r, c, grid):
        self.rc = CellRC(r, c)
        self.grid = grid

    def up(self):
        self.rc.r -= 1
        self.rc.r %= self.grid.grid_rows
        print(self.rc.r)
        pass

    def down(self):
        self.rc.r += 1
        self.rc.r %= self.grid.grid_rows
        pass

    def left(self):
        self.rc.c -= 1
        self.rc.c %= self.grid.grid_cols
        pass

    def right(self):
        self.rc.c += 1
        self.rc.c %= self.grid.grid_cols
        pass


class GridWidget(QWidget):
    def __init__(self, parent, controlWidget):
        super().__init__()

        self.parent = parent
        self.control_widget = controlWidget

        self.grid_rows = 50
        self.grid_cols = 50
        self.tile_size_x = 10
        self.tile_size_y = self.tile_size_x

        self.grid = np.ones((self.grid_rows, self.grid_cols, 3), dtype=np.uint)*255

        self.setMouseTracking(True)
        self.x_prev = -1
        self.y_prev = -1

        self.activeCell = None
        self.key_r_down = False
        self.key_c_down = False

        self.setMinimumSize(self.grid_cols*self.tile_size_x, self.grid_rows*self.tile_size_y)


    def from_image(self, img_path):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.grid_rows, self.grid_cols))
        self.x_prev = -1
        self.y_prev = -1

        self.activeCell = None
        self.grid = img[:, :, (2, 1, 0)]
        self.repaint()


    def from_grid(self, grid: Grid):
        self.x_prev = -1
        self.y_prev = -1
        self.activeCell = None

        self.grid_rows = grid.get_range("y").size()
        self.grid_cols = grid.get_range("x").size()

        self.grid = grid.get_filled().copy()
        #self.setMinimumSize(self.grid_cols * self.tile_size_x, self.grid_rows * self.tile_size_y)
        self.repaint()


    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_grid(event, qp)
        self.draw_activeCell(qp)
        qp.end()


    def keyPressEvent(self, event):
        super(GridWidget, self).keyPressEvent(event)
        if (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_S:
            print("export")
            self.export("test_input_tmp.txt")

        if self.activeCell:
            if event.key() == Qt.Key_Up:
                self.activeCell.up()
            if event.key() == Qt.Key_Down:
                self.activeCell.down()
            if event.key() == Qt.Key_Left:
                self.activeCell.left()
            if event.key() == Qt.Key_Right:
                self.activeCell.right()
            self.repaint()

        if event.key() == Qt.Key_R:
            self.key_r_down = True
        if event.key() == Qt.Key_C:
            self.key_c_down = True


    def keyReleaseEvent(self, event):
        self.key_r_down = False
        self.key_c_down = False


    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        if event.buttons() == Qt.LeftButton:
            self.activeCell = None
            if self.key_r_down:
                self.fillRow(x, y)
            elif self.key_c_down:
                self.fillColumn(x, y)
            else:
                self.fillCell(x, y)
            self.repaint()
        if event.buttons() == Qt.RightButton:
            co = self.get_cell_coords(x, y)
            if co:
                self.activeCell = ActiveCell(co.r, co.c, self)
                self.repaint()


    def mouseReleaseEvent(self, event):
        self.x_prev = -1
        self.y_prev = -1


    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            if not (int(x / self.tile_size_x) - int(self.x_prev / self.tile_size_x) == 0 and \
                int(y / self.tile_size_y) - int(self.y_prev / self.tile_size_y) == 0):
                self.fillCell(x, y)
                self.repaint()
            self.x_prev = x
            self.y_prev = y


    def get_cell_coords(self, x, y):
        pc = int(x / self.tile_size_x)
        pr = int(y / self.tile_size_y)
        if pr >= 0 and pr < self.grid_rows and pc >= 0 and pc < self.grid_cols:
            return CellRC(pr, pc)
        else:
            return None


    def fillCell(self, x, y):
        co = self.get_cell_coords(x,y)
        if co:
            self.grid[co.r][co.c] = self.control_widget.color_button.color_rgb()
            self.parent.update_output_canvas()


    def fillRow(self, x, y):
        co = self.get_cell_coords(x,y)
        if co:
            for c in range(self.grid_cols):
                self.grid[co.r][c] = self.control_widget.color_button.color_rgb()
            self.parent.update_output_canvas()


    def fillColumn(self, x, y):
        co = self.get_cell_coords(x,y)
        if co:
            for r in range(self.grid_rows):
                self.grid[r][co.c] = self.control_widget.color_button.color_rgb()
            self.parent.update_output_canvas()


    def draw_activeCell(self, qp):
        if self.activeCell:
            r = self.activeCell.rc.r
            c = self.activeCell.rc.c
            self.grid[r][c] = 255 - self.grid[r][c]
            qp.setBrush(QColor(self.grid[r][c][0], self.grid[r][c][1], self.grid[r][c][2]))
            qp.drawRect(c * self.tile_size_x, r * self.tile_size_y, self.tile_size_x, self.tile_size_y)


    def draw_grid(self, event, qp):
        qp.setPen(QColor(0, 0, 0))

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                qp.setBrush(QColor(self.grid[r][c][0],self.grid[r][c][1], self.grid[r][c][2]))
                qp.drawRect(c*self.tile_size_x, r*self.tile_size_y, self.tile_size_x, self.tile_size_y)


    def to_grid(self):
        xmin = self.control_widget.spin_xmin.value()
        xdelta = self.control_widget.spin_xdelta.value()
        ymin = self.control_widget.spin_ymin.value()
        ydelta = self.control_widget.spin_ydelta.value()

        xrange = Range(xmin, xmin+xdelta, self.grid_cols)
        yrange = Range(ymin, ymin+ydelta, self.grid_rows)
        filled = np.zeros_like(self.grid, dtype=np.uint)
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                filled[r][c] = [self.grid[r][c][0], self.grid[r][c][1], self.grid[r][c][2]]
        grid = Grid(xrange, yrange, filled=filled)
        return grid


    def export(self, filename):
        ## export x, y ranges and grid
        with open(filename, "w") as f:
            f.write("x:-6, 6.25, "+ str(self.grid_cols) +"\n")
            f.write("y:0,2, "+ str(self.grid_rows) + "\n")
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    f.write(str(self.grid[r][c][0]) + "," + str(self.grid[r][c][1]) + "," + str(self.grid[r][c][2]))
                    if c<self.grid_cols-1:
                        f.write(",")
                f.write("\n")


class QColorButton(QPushButton):

    def __init__(self, *args, **kwargs):
        super(QColorButton, self).__init__(*args, **kwargs)

        self.color = "#FFFFFF"
        self.setMaximumWidth(64)
        self.pressed.connect(self.onColorPicker)

    def set_color(self, color):
        if color != self.color:
            self.color = color

        if self.color:
            self.setStyleSheet("background-color: %s;" % self.color)
        else:
            self.setStyleSheet("")

    def color(self):
        return self.color

    def color_rgb(self):
        c = QColor(self.color)
        return [c.red(), c.green(), c.blue()]

    def onColorPicker(self):
        dlg = QColorDialog(self)
        if self.color:
            dlg.setCurrentColor(QColor(self.color))

        if dlg.exec_():
            self.set_color(dlg.currentColor().name())

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.set_color(None)

        return super(QColorButton, self).mousePressEvent(e)


class ControlWidget(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        self.t_funcs = TFunctions()
        self.cur_t_func = TMatrix.create_identity_map()

        self.ctrl1_widget = QWidget()

        self.color_button = QColorButton()

        self.check_lines = QCheckBox("Draw grid")
        self.check_lines.setChecked(False)
        self.check_lines.stateChanged.connect(lambda: self.check_btn_state(self.check_lines))

        self.combo_func = QComboBox()
        for t_func in self.t_funcs.f:
            self.combo_func.addItem(t_func.get("name"))

        self.param_sliders = []

        self.combo_func.activated[str].connect(self.onComboActivated)

        self.layout_c1 = QFormLayout()
        self.layout_c1.addRow("Color: ", self.color_button)
        self.layout_c1.addRow("Do:", self.check_lines)
        self.layout_c1.addRow("Func:", self.combo_func)

        self.ctrl1_widget.setLayout(self.layout_c1)


        self.ctrl2_widget = QWidget()

        self.layout_c2 = QFormLayout()
        self.spin_xmin = QSpinBox()
        self.spin_xmin.setRange(-30, 30)
        self.spin_xmin.setValue(-3)
        self.spin_xmin.valueChanged[int].connect(lambda x: self.update_output_canvas())
        self.layout_c2.addRow("x min: ", self.spin_xmin)
        self.spin_xdelta = QSpinBox()
        self.spin_xdelta.setRange(1, 100)
        self.spin_xdelta.setValue(6)
        self.spin_xdelta.valueChanged[int].connect(lambda x: self.update_output_canvas())
        self.layout_c2.addRow("x delta: ", self.spin_xdelta)
        self.spin_ymin = QSpinBox()
        self.spin_ymin.setRange(-30, 30)
        self.spin_ymin.setValue(0)
        self.spin_ymin.valueChanged[int].connect(lambda x: self.update_output_canvas())
        self.layout_c2.addRow("y min: ", self.spin_ymin)
        self.spin_ydelta = QSpinBox()
        self.spin_ydelta.setRange(1, 100)
        self.spin_ydelta.setValue(2)
        self.spin_ydelta.valueChanged[int].connect(lambda x: self.update_output_canvas())
        self.layout_c2.addRow("y delta: ", self.spin_ydelta)

        self.ctrl2_widget.setLayout(self.layout_c2)


        self.layout = QVBoxLayout()

        self.tabs_widget = QTabWidget()
        self.tabs_widget.addTab(self.ctrl1_widget, "Function")
        self.tabs_widget.addTab(self.ctrl2_widget, "Panel")
        self.layout.addWidget(self.tabs_widget)
        self.setLayout(self.layout)


    def onComboActivated(self, name):
        for slider in self.param_sliders:
            self.layout_c1.removeRow(slider)
        self.param_sliders = []
        args = self.t_funcs.get_f_args_by_name(name)
        if args:
            for arg in args:
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(100)
                slider.setValue(50)
                slider.setTickInterval(1)
                slider.valueChanged[int].connect(self.slider_val_changed)
                self.param_sliders.append(slider)
                self.layout_c1.addRow(arg[0], slider)
        self.update_output_canvas()


    def slider_val_changed(self, value):
        print("value=", value)
        self.update_output_canvas()


    def update_output_canvas(self):
        func_name = self.combo_func.currentText()
        self.cur_t_func = self.compose_func(func_name)
        self.parent.update_output_canvas()


    def compose_func(self, f_name):
        func = self.t_funcs.get_f_func_by_name(f_name)
        args = self.t_funcs.get_f_args_by_name(f_name)
        f_inputs = []
        if args:
            for i,arg in enumerate(args):
                v_min = arg[1]
                v_max = arg[2]
                v = v_min + self.param_sliders[i].value()/100 * (v_max - v_min)
                f_inputs.append(v)
        return func(f_inputs)


    def check_btn_state(self, b):
        if b.text() == "Draw grid":
            self.parent.update_output_canvas()



class OutputWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.label = QLabel(self)
        self.pixmap = QPixmap()
        self.label.setPixmap(self.pixmap)

    def update_pixmap(self, np_arr):
        width, height, ch = np_arr.shape
        bytes_per_line = ch*width
        qimage = QImage(np_arr.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimage))
        self.setMinimumSize(qimage.width(), qimage.height())


class MainWidget(QWidget):
    def __init__(self, img_file):
        super().__init__()
        self.initUI(img_file)


    def initUI(self, img_file):
        self.control_widget = ControlWidget(self)

        self.grid_widget_src = GridWidget(self, self.control_widget)
        self.grid_widget_src.from_image(img_file)

        self.output_widget = OutputWidget()
        self.update_output_canvas()

        self.left_arrow_label = QLabel("  >>  ")
        font = QFont()
        font.setPixelSize(25)
        self.left_arrow_label.setFont(font)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.control_widget)
        self.layout.addWidget(self.grid_widget_src)
        self.layout.addWidget(self.left_arrow_label)
        self.layout.addWidget(self.output_widget)

        self.setLayout(self.layout)

        self.setWindowTitle('CTransformEditor')

        self.show()


    def keyPressEvent(self, event):
        self.grid_widget_src.keyPressEvent(event)


    def keyReleaseEvent(self, event):
        self.grid_widget_src.keyReleaseEvent(event)


    def update_output_canvas(self):
        t_grid = self.grid_widget_src.to_grid().transform_by(self.control_widget.cur_t_func)
        canvas = t_grid.draw_on_canvas(do_fill=True,
                                       do_outlines=self.control_widget.check_lines.isChecked())
        self.output_widget.update_pixmap(canvas)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWidget("test.png")
    sys.exit(app.exec_())


