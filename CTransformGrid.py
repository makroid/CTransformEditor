import numpy as np
import typing
from typing import Iterable
import sys
import cv2


class Range:
    def __init__(self, min=0, max=1, len=100):
        assert(min <= max)
        assert(len > 0)
        self.min = min
        self.max = max
        self.len = len

    def to_np(self):
        return np.linspace(self.min, self.max, self.len)

    def size(self):
        return self.len



## Cayley map: K(z) = (z-i)/(z+i)
def _cayley_single(z):
    return (z-1j)/(z+1j)


def _cayley_list(z:Iterable[complex]):
    return [(zz-1j)/(zz+1j) for zz in z]


def cayley(z):
    if type(z)=="list":
        return _cayley_list(z)
    else:
        return _cayley_single(z)


def get_bounds(vs:Iterable[complex]):

    min_x = min_y = sys.maxsize
    max_x = max_y = -sys.maxsize

    for a in vs:
        if a.real < min_x:
            min_x = a.real
        if a.real > max_x:
            max_x = a.real
        if a.imag < min_y:
            min_y = a.imag
        if a.imag > max_y:
            max_y = a.imag
    return { 'min_x':min_x, "max_x":max_x, "min_y":min_y, "max_y":max_y }


def scale_in_range(vs, min_x, max_x, min_y, max_y):
    b = get_bounds(vs)
    scale_x = (max_x - min_x) / (b['max_x'] - b['min_x'])
    scale_y = (max_y - min_y) / (b['max_y'] - b['min_y'])

    sv = []

    for cvv in vs:
        sv_x = int((cvv.real + abs(b['min_x'])) * scale_x)
        sv_y = int((cvv.imag + abs(b['min_y'])) * scale_y)
        sv.append(sv_x + sv_y*1j)
    return sv


class Grid:

    def __init__(self, xrange:Range, yrange:Range, points=None, filled=None):
        self._xrange    = xrange
        self._yrange    = yrange
        if points is None:
            self._points = np.array([[i + k * 1j for i in xrange.to_np()] for k in yrange.to_np()])
        else:
            self._points = points
        if filled is None:
            ## add color channel and decrease xy dim by one
            self._filled = np.ones(np.subtract(self._points.shape + (3,), (1,1,0))) * 128
        else:
            self._filled = filled


    @staticmethod
    def from_file(filename):
        import re
        with open(filename, "r") as f:
            xrange = None
            yrange = None
            lines = f.readlines()

            hit_x = re.search(r'\s*x:\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)', lines[0], re.M | re.I)
            if hit_x:
                xrange = Range(float(hit_x.group(1)), float(hit_x.group(2)), int(hit_x.group(3)))
                print("min=", float(hit_x.group(1)))
                print("max=", float(hit_x.group(2)))
                print("len=", int(hit_x.group(3)))

            hit_y = re.search(r'\s*y:\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)', lines[1], re.M | re.I)
            if hit_y:
                yrange = Range(float(hit_y.group(1)), float(hit_y.group(2)), int(hit_y.group(3)))
                print("min=", float(hit_y.group(1)))
                print("max=", float(hit_y.group(2)))
                print("len=", int(hit_y.group(3)))

            filled = np.zeros((yrange.len, xrange.len, 3))
            for i in range(2, len(lines)):
                filled[i-2,:,:] = np.array([float(id) for id in lines[i].split(",")]).reshape(xrange.len, 3)

            if xrange is None or yrange is None:
                sys.exit("invalid file")

            return Grid(xrange, yrange, filled=filled)


    def get_points(self):
        return self._points.ravel()


    def get_filled(self):
        return self._filled


    def get_range(self, dir):
        if dir=="x":
            return self._xrange
        else:
            return self._yrange


    def transform_by(self, M):
        cp = np.ones(self._points.shape) * (0+0j)
        for i in range(self._yrange.len):
            for j in range(self._xrange.len):
                cp[i][j] = M(self._points[i][j])
            #print("=>", p, "->", cp[i])
        b = get_bounds(cp.ravel())
        xrange = Range(b['min_x'], b['max_x'], self._xrange.len)
        yrange = Range(b['min_y'], b['max_y'], self._yrange.len)
        return Grid(xrange, yrange, cp, self._filled)


    def transform_rescale(self, min_x, max_x, min_y, max_y):
        cur_xmin = self._xrange.min
        cur_xmax = self._xrange.max
        cur_ymin = self._yrange.min
        cur_ymax = self._yrange.max
        scale_x = (max_x - min_x) / (cur_xmax - cur_xmin)
        scale_y = (max_y - min_y) / (cur_ymax - cur_ymin)

        points = np.ones(self._points.shape) * (0+0j)
        for i in range(self._yrange.len):
            for j in range(self._xrange.len):
                p = self._points[i][j]
                real = int((p.real-cur_xmin) * scale_x + min_x)
                imag = int((p.imag-cur_ymin) * scale_y + min_y)
                points[i][j] = real + imag*1j
        xrange = Range(min_x, max_x, self._xrange.len)
        yrange = Range(min_y, max_y, self._yrange.len)

        return Grid(xrange, yrange, points, self._filled)


    def draw_on_canvas(self, canvas_w=500, canvas_h=500, do_fill=True, do_outlines=False):
        my_grid = self.transform_rescale(0, canvas_w - 1, 0, canvas_h - 1)
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40

        white = (255,) * 3
        black = (0,) * 3

        if do_fill:
            for cell in Grid.CellIter(my_grid):
                pts = np.array([[c.real, c.imag] for c in cell[0]])
                col = cell[1][::-1]
                cv2.fillPoly(canvas, np.int32([pts]), color=(int(col[0]), int(col[1]), int(col[2])))

        if do_outlines:
            ## draw cayley transformed points
            for point in my_grid.get_points():
                cv2.circle(canvas, (int(point.real), int(point.imag)), 2, white, -1)

            ## draw gridlines
            for row in Grid.RowIter(my_grid):
                for i in range(1, len(row)):
                    p1x = int(row[i - 1].real)
                    p1y = int(row[i - 1].imag)
                    p2x = int(row[i].real)
                    p2y = int(row[i].imag)
                    cv2.line(canvas, (p1x, p1y), (p2x, p2y), black, 1)

            for col in Grid.ColIter(my_grid):
                for i in range(1, len(col)):
                    p1x = int(col[i - 1].real)
                    p1y = int(col[i - 1].imag)
                    p2x = int(col[i].real)
                    p2y = int(col[i].imag)
                    cv2.line(canvas, (p1x, p1y), (p2x, p2y), black, 1)
        canvas = canvas[:,:,(2,1,0)].copy()
        return canvas


    class RowIter:
        def __init__(self, parent):
            self._cur_row = 0
            self._parent = parent
            self._nrows = parent._yrange.size()
            self._ncols = parent._xrange.size()

        def __iter__(self):
            return self

        def __next__(self):
            if self._cur_row >= self._nrows:
                raise StopIteration
            else:
                self._cur_row += 1
                return self._parent._points[self._cur_row-1]


    class ColIter:
        def __init__(self, parent):
            self._cur_col = 0
            self._parent = parent
            self._nrows = parent._yrange.size()
            self._ncols = parent._xrange.size()

        def __iter__(self):
            return self

        def __next__(self):
            if self._cur_col >= self._ncols:
                raise StopIteration
            else:
                idx = [[i for i in range(self._nrows)], [self._cur_col for i in range(self._nrows)]]
                self._cur_col += 1
                return self._parent._points[idx]


    class CellIter:
        def __init__(self, parent):
            self._parent = parent
            self._nrows = parent._yrange.size()
            self._ncols = parent._xrange.size()
            self._cur_row = 0
            self._cur_col = -1
            self._niter = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._cur_col += 1

            if self._cur_col == self._ncols-1:
                self._cur_col = 0
                self._cur_row += 1

            self._niter += 1

            if self._niter >= (self._nrows-1)*(self._ncols-1):
                raise StopIteration

            idx = [[self._cur_row]*2 +  [self._cur_row+1]*2,  # row idx
                   [self._cur_col, self._cur_col+1, self._cur_col+1, self._cur_col]]     # col idx
            return (self._parent._points[idx], self._parent._filled[self._cur_row, self._cur_col, :])


class TMatrix:
    def __init__(self, a, b, c, d):
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def __call__(self, z:complex) -> complex:
        return (self._a*z + self._b)/(self._c*z + self._d)

    @staticmethod
    def create_identity_map(p=[]):
        return TMatrix(1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j)

    @staticmethod
    def create_hyperbolic_map(p):
        u = p[0]
        import cmath
        assert(u>1)
        return TMatrix(u+0j, cmath.sqrt(u*u-1)+0j, cmath.sqrt(u*u-1)+0j, u+0j)

    @staticmethod
    def create_rotation_map(p):
        theta = p[0]
        import cmath
        return TMatrix(cmath.exp(theta*1j), 0, 0, cmath.exp(-theta*1j))

    @staticmethod
    def create_cayley_map(p=[]):
        return TMatrix(1, -1j, 1, 1j)

    @staticmethod
    def create_moebius_map(p):
        a = complex(p[0], p[1])
        b = complex(p[2], p[3])
        c = complex(p[4], p[5])
        d = complex(p[6], p[7])
        return TMatrix(a, b, c, d)


class CQuadraticFunc:

    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c

    def __call__(self, z: complex) -> complex:
        return self._a * z * z + self._b * z + self._c

    @staticmethod
    def create(p):
        a = complex(p[0], p[1])
        b = complex(p[2], p[3])
        c = complex(p[4], p[5])
        return CQuadraticFunc(a, b, c)


class CExponentialFunc:

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __call__(self, z: complex) -> complex:
        return self._a * z ** self._b

    @staticmethod
    def create(p):
        a = complex(p[0], p[1])
        b = complex(p[2], p[3])
        return CExponentialFunc(a, b)


class CLinFunc:

    def __init__(self, func=lambda z:z, a=complex(1,0), b=complex(0,0)):
        self._a = a
        self._b = b
        self.func = func

    def __call__(self, z:complex) -> complex:
        return self.func(self._a*z) + self._b



class TFunctions:

    def __init__(self):

        self.f = [
            {
                "name": "identity_map",
                "args": None,
                "func": TMatrix.create_identity_map
            },
            {
                "name" : "caylay_map",
                "args" : None,
                "func" : TMatrix.create_cayley_map
            },
            {
                "name" : "rotation_map",
                "args" : [["angle", 0, 360, 0, 1]],
                "func" : TMatrix.create_rotation_map
            },
            {
                "name" : "hyperbolic_map",
                "args" : [["p", 1, 2, 1, 0.001]],
                "func" : TMatrix.create_hyperbolic_map
            },
            {
                "name": "quadratic_map",
                "args": [["a_r", -2, 4, 1, 0.1],
                         ["a_i", -2, 4, 0, 0.1],
                         ["b_r", -2, 4, 0, 0.1],
                         ["b_i", -2, 4, 0, 0.1],
                         ["c_r", -2, 4, 0, 0.1],
                         ["c_i", -2, 4, 0, 0.1]],
                "func": CQuadraticFunc.create
            },
            {
                "name": "exponential_map",
                "args": [["a_r", -2, 4, 1, 0.1],
                         ["a_i", -2, 4, 0, 0.1],
                         ["b_r", -2, 4, 1, 0.1],
                         ["b_i", -2, 4, 0, 0.1]],
                "func": CExponentialFunc.create
            },
            {
                "name" : "moebius_map",
                "args" : [["a_r", -2, 4, 1, 0.1],
                          ["a_i", -2, 2, 0, 0.1],
                          ["b_r", -2, 2, 0, 0.1],
                          ["b_i", -2, 2, 0, 0.1],
                          ["c_r", -2, 2, 0, 0.1],
                          ["c_i", -2, 2, 0, 0.1],
                          ["d_r", -2, 4, 0, 0.1],
                          ["d_i", -2, 2, 0, 0.1],
                          ],
                "func" : TMatrix.create_moebius_map
            }
        ]

    def get_f_args_by_name(self, name):
        for f in self.f:
            if f["name"] == name:
                return f["args"]
        return None

    def get_f_func_by_name(self, name):
        for f in self.f:
            if f["name"] == name:
                return f["func"]
        return None
