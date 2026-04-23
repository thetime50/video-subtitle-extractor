
from typing import Union
from dataclasses import dataclass
from shapely.geometry import Polygon

@dataclass
class SubtitleArea:
    """
    字幕区域
    """
    ymin: Union[int, float]
    ymax: Union[int, float]
    xmin: Union[int, float]
    xmax: Union[int, float]
    # 字幕区域在视频中的位置
    ab_section: range = None
    
    def __init__(self, ymin: Union[int, float], ymax: Union[int, float], 
                 xmin: Union[int, float], xmax: Union[int, float], 
                 ab_section: range = None,
                 fymin: Union[int, float] = None,
                 fymax: Union[int, float] = None,
                 fxmin: Union[int, float] = None,
                 fxmax: Union[int, float] = None):
        self.ymin = ymin
        self.ymax = ymax    
        self.xmin = xmin
        self.xmax = xmax
        self.ab_section = ab_section

        # 字体大小参考框坐标 相对字幕区域
        self.fymin = fymin
        self.fymax = fymax
        self.fxmin = fxmin
        self.fxmax = fxmax

    def normalized(self):
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin

    def is_empty(self):
        return self.xmin == 0 and self.xmax == 0 and self.ymin == 0 and self.ymax == 0

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def in_ab_section(self, frame_idx):
        return True

    def to_polygon(self):
        return Polygon([[self.xmin, self.ymin], [self.xmax, self.ymin], [self.xmax, self.ymax], [self.xmin, self.ymax]])
    
    def clone(self):
        return SubtitleArea(self.ymin, self.ymax, self.xmin, self.xmax, self.ab_section,
            self.fymin, self.fymax, self.fxmin, self.fxmax)
    
    # 字体大小参考框逻辑
    @property
    def fwidth(self):
        return None if self.fxmin is None and self.fxmax is None else self.fxmax - self.fxmin

    @property
    def fheight(self):
        return None if self.fymin is None and self.fymax is None else self.fymax - self.fymin
    

    @property
    def fxmin_g(self):
        return None if self.fxmin is None else self.fxmin + self.xmin

    @property
    def fxmax_g(self):
        return None if self.fxmax is None else self.fxmax + self.xmin

    @property
    def fymin_g(self):
        return None if self.fymin is None else self.fymin + self.ymin

    @property
    def fymax_g(self):
        return None if self.fymax is None else self.fymax + self.ymin

    def normalized_font_rect(self):
        if self.fxmin is None or self.fxmax is None or self.fymin is None or self.fymax is None:
            return
        if self.fxmin > self.fxmax:
            self.fxmin, self.fxmax = self.fxmax, self.fxmin
        if self.fymin > self.fymax:
            self.fymin, self.fymax = self.fymax, self.fymin

        if self.fxmin < 0:
            self.fxmin = 0
        if self.fymin < 0:
            self.fymin = 0
        if self.fxmax > self.width:
            self.fxmax = self.width
        if self.fymax > self.height:
            self.fymax = self.height
        
        if self.fwidth <=0:
            self.fxmin = 0
            self.fxmax = 0.005
        if self.fheight <=0:
            self.fymin = 0
            self.fymax = 0.005
