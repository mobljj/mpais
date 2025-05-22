# -*- coding: utf-8 -*-
import sys
import os
from io import BytesIO
import qdarkstyle

def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        # PyInstaller 创建临时文件夹并将路径存储在 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # 如果不是通过 PyInstaller 运行（开发环境），使用当前文件所在目录
        base_path = os.path.abspath(".") # 或者 os.path.dirname(__file__)

    return os.path.join(base_path, relative_path)

os.environ["QT_API"] = "pyside6"
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QGraphicsView, QGraphicsScene, QFrame, QStatusBar, QMenuBar, QMenu, # 添加了 QMenu
    QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QGroupBox, QProgressBar, QAbstractItemView
)
from qtpy.QtGui import (
    QPixmap, QPainter, QPen, QAction, QImageReader, QGuiApplication,
    QImage, QCursor, QTransform, QColor, QFont, QIcon
)
from qtpy.QtCore import Qt, QRectF, QPointF, Signal, QThread, QObject, Slot

# 处理和绘图所需的库
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg') # 设置 Matplotlib 后端为 'Agg'，用于非交互式绘图
import matplotlib.pyplot as plt
try:
    # 设置 Matplotlib 字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"警告: 未能成功设置 Matplotlib 中文字体 - {e}")


# --- WorkerSignals 和 Worker 类 ---
class WorkerSignals(QObject):
    """
    定义工作线程可以发出的信号。
    """
    finished = Signal() # 任务完成信号
    error = Signal(tuple) # 任务出错信号，传递错误信息
    result = Signal(object) # 任务成功返回结果信号
    progress = Signal(int) # 任务进度信号
    status_update = Signal(str) # 状态更新信号
    # 发出：任务名称, 结果图像 (QPixmap), 源图像路径 (用于模板匹配, 坐标提取)
    new_image_result = Signal(str, QPixmap, str)
    # 发出：绘图结果 (QPixmap) (全局绘图)
    new_plot_result = Signal(QPixmap)

class Worker(QObject):
    """
    通用工作线程类，用于在单独的线程中执行耗时任务。
    """
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function # 要执行的函数
        self.args = args # 函数的位置参数
        self.kwargs = kwargs # 函数的关键字参数
        self.signals = WorkerSignals() # 创建信号对象

    @Slot()
    def run(self):
        """
        执行任务函数。
        """
        try:
            # 检查函数是否接受 'signals' 参数，如果接受则传入
            func_params = self.function.__code__.co_varnames[:self.function.__code__.co_argcount]
            if 'signals' in func_params:
                self.kwargs['signals'] = self.signals
            result = self.function(*self.args, **self.kwargs) # 执行函数
        except Exception as e:
            import traceback # 导入 traceback 模块以获取详细的错误信息
            print(f"工作线程错误 (函数 '{self.function.__name__}'): {e}\n{traceback.format_exc()}")
            exctype, value = sys.exc_info()[:2] # 获取异常类型和值
            self.signals.error.emit((exctype, value, traceback.format_exc())) # 发出错误信号
        else:
            self.signals.result.emit(result) # 发出结果信号
        finally:
            self.signals.finished.emit() # 发出完成信号


# --- InteractiveGraphicsView 类 ---
class InteractiveGraphicsView(QGraphicsView):
    """
    可交互的图形视图，支持缩放、平移、区域选择和点选。
    """
    regionSelected = Signal(QRectF) # 区域选择信号
    pointClicked = Signal(QPointF) # 点击信号

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing) # 设置抗锯齿渲染
        self.setRenderHint(QPainter.SmoothPixmapTransform) # 设置平滑的Pixmap变换
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) # 设置变换锚点为鼠标下方
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse) # 设置缩放锚点为鼠标下方
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 关闭垂直滚动条
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 关闭水平滚动条
        self.setDragMode(QGraphicsView.ScrollHandDrag) # 设置默认拖动模式为手型拖动
        self.start_point = QPointF() # 区域选择的起始点
        self.selecting_region = False # 是否正在选择区域
        self.selecting_point = False # 是否正在选择点
        self._is_panning = False # 是否正在平移
        self._last_pan_point = QPointF() # 上一个平移点

    def set_selecting_region_mode(self, enabled):
        """设置区域选择模式。"""
        self.selecting_region = enabled
        self.selecting_point = False
        if enabled:
            self.setDragMode(QGraphicsView.RubberBandDrag) # 设置拖动模式为橡皮筋选择
            self.setCursor(Qt.CrossCursor) # 设置光标为十字形
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag) # 恢复为手型拖动
            self.setCursor(Qt.ArrowCursor) # 恢复为箭头光标

    def set_selecting_point_mode(self, enabled):
        """设置点选择模式。"""
        self.selecting_point = enabled
        self.selecting_region = False
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag) # 设置无拖动模式
            self.setCursor(Qt.PointingHandCursor) # 设置光标为指示手型
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag) # 恢复为手型拖动
            self.setCursor(Qt.ArrowCursor) # 恢复为箭头光标

    def wheelEvent(self, event):
        """处理鼠标滚轮事件以进行缩放。"""
        if not self.scene() or not self.scene().items():
            return
        zoom_in_factor = 1.15 # 放大因子
        zoom_out_factor = 1 / zoom_in_factor # 缩小因子
        old_pos = self.mapToScene(event.position().toPoint()) # 获取鼠标在场景中的旧位置
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor # 根据滚轮方向确定缩放因子
        current_scale = self.transform().m11() # 获取当前缩放比例
        # 限制缩放范围
        if current_scale * zoom_factor < 0.01 or current_scale * zoom_factor > 100:
            return
        self.scale(zoom_factor, zoom_factor) # 应用缩放
        new_pos = self.mapToScene(event.position().toPoint()) # 获取鼠标在场景中的新位置
        delta = new_pos - old_pos # 计算位置差
        self.translate(delta.x(), delta.y()) # 平移视图以保持鼠标位置不变

    def mousePressEvent(self, event):
        """处理鼠标按下事件。"""
        if self.selecting_region and event.button() == Qt.LeftButton:
            # 如果是区域选择模式且按下左键，记录起始点
            self.start_point = self.mapToScene(event.pos())
            super().mousePressEvent(event)
        elif self.selecting_point and event.button() == Qt.LeftButton:
            # 如果是点选择模式且按下左键，发出点击信号
            scene_pos = self.mapToScene(event.pos())
            self.pointClicked.emit(scene_pos)
        elif (event.button() == Qt.MiddleButton or
              (event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier)):
            # 如果按下中键或 Ctrl+左键，且不是选择模式，则开始平移
            if not self.selecting_region and not self.selecting_point:
                self._is_panning = True
                self._last_pan_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor) # 设置光标为闭合手型
                event.accept()
            else:
                event.ignore()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件。"""
        if self._is_panning:
            # 如果正在平移，则移动滚动条
            delta = event.pos() - self._last_pan_point
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            self._last_pan_point = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件。"""
        button_match = event.button() == Qt.MiddleButton or \
                       (event.button() == Qt.LeftButton and self._is_panning)
        if self._is_panning and button_match:
            # 如果正在平移且释放了相应按键，则停止平移
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor if not self.selecting_region and not self.selecting_point else self.cursor())
            event.accept()
        elif self.selecting_region and event.button() == Qt.LeftButton:
            # 如果是区域选择模式且释放左键，发出区域选择信号
            end_point = self.mapToScene(event.pos())
            selected_rect = QRectF(self.start_point, end_point).normalized()
            if selected_rect.isValid():
                self.regionSelected.emit(selected_rect)
            super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

    def fit_view_to_scene(self):
        """调整视图以适应场景内容。"""
        if self.scene() and self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def set_pixmap(self, pixmap):
        """设置显示的图像。"""
        if self.scene():
            self.scene().clear() # 清空场景
            if pixmap and not pixmap.isNull():
                self.scene().addPixmap(pixmap) # 添加图像
                self.scene().setSceneRect(QRectF(pixmap.rect())) # 设置场景矩形
                self.fit_view_to_scene() # 适应视图
            else:
                self.scene().addText("无图像或结果") # 显示提示信息
                self.scene().setSceneRect(QRectF(0, 0, 200, 100))


# --- ResultDisplayWindow 类 ---
class ResultDisplayWindow(QWidget):
    """
    用于显示图像结果的弹出窗口。
    """
    windowClosed = Signal(str)  # 窗口关闭信号，发出窗口类型 ('tm', 'ce', 'plot')

    # 添加 icon_rel_path 参数
    def __init__(self, window_type, title="结果显示", icon_rel_path=None, parent=None):
        super().__init__(parent)
        self.window_type = window_type # 窗口类型
        self.setWindowTitle(title) # 设置窗口标题
        self.setGeometry(200, 200, 800, 600)  # 可以按需调整默认大小/位置

        # --- 设置窗口图标 ---
        if icon_rel_path:
            try:
                icon_path_actual = resource_path(icon_rel_path)  # 使用辅助函数获取绝对路径
                if os.path.exists(icon_path_actual):
                    self.setWindowIcon(QIcon(icon_path_actual))
                else:
                    print(f"警告: 弹出窗口图标文件未找到: {icon_path_actual}")
            except Exception as e:
                print(f"错误: 加载弹出窗口图标时出错 ({icon_rel_path}): {e}")
        # --- 图标设置结束 ---

        layout = QVBoxLayout(self) # 创建垂直布局
        self.scene = QGraphicsScene() # 创建图形场景
        self.view = InteractiveGraphicsView(self.scene)  # 假设这个类定义在别处
        layout.addWidget(self.view) # 将视图添加到布局
        self._source_image_path = None # 源图像路径

    def set_image(self, pixmap, source_image_path=None):
        """设置窗口中显示的图像。"""
        self.view.set_pixmap(pixmap)  # 假设 view 有 set_pixmap 方法
        self._source_image_path = source_image_path # 记录源图像路径

    def get_source_path(self):
        """获取源图像路径。"""
        return self._source_image_path

    def closeEvent(self, event):
        """处理窗口关闭事件。"""
        self.windowClosed.emit(self.window_type) # 发出窗口关闭信号
        event.accept()


# --- 用于排序点的辅助函数 ---
def sort_grid_points(points, row_tolerance=50, col_tolerance=50):
    """
    对网格点进行排序，首先按行，然后按列。
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if points.shape[0] == 0:
        return points

    # 按 Y 坐标对点进行排序
    sorted_y_indices = np.argsort(points[:, 1])
    sorted_y_points = points[sorted_y_indices]

    row_clusters_indices = []
    if sorted_y_points.shape[0] > 0:
        current_cluster = [sorted_y_indices[0]]
        last_y = sorted_y_points[0, 1]
        # 根据 Y 坐标的容差将点分组成行
        for i in range(1, len(sorted_y_points)):
            current_y = sorted_y_points[i, 1]
            if abs(current_y - last_y) < row_tolerance:
                current_cluster.append(sorted_y_indices[i])
            else:
                row_clusters_indices.append(current_cluster)
                current_cluster = [sorted_y_indices[i]]
                last_y = current_y
        row_clusters_indices.append(current_cluster) # 添加最后一个聚类

    sorted_points_list = []
    # 按行的平均 Y 坐标对行聚类进行排序
    row_clusters_indices.sort(key=lambda indices: np.mean(points[indices, 1]) if indices else 0)

    # 在每行内按 X 坐标对点进行排序
    for row_indices in row_clusters_indices:
        if not row_indices:
            continue
        row_points_subset = points[row_indices]
        sorted_x_in_row_indices = np.argsort(row_points_subset[:, 0])
        original_indices_sorted_by_x = np.array(row_indices)[sorted_x_in_row_indices]
        sorted_points_list.extend(points[original_indices_sorted_by_x])
    return np.array(sorted_points_list)


# --- 主应用程序窗口 ---
class MainWindow(QMainWindow):
    """
    应用程序的主窗口。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("相似模拟实验监测点自动识别系统") # 设置窗口标题
        self.setGeometry(100, 100, 1000, 700) # 设置窗口几何形状
        self.image_files = {} # 存储加载的图像文件路径 {显示名称: 路径}
        self.current_image_path = None # 当前选中的图像路径
        self.processing_results = {} # 存储处理结果 {图像路径: {结果类型: 结果}}
        self.threadpool = [] # 存储活动的工作线程和工作对象
        # 更新的结果窗口字典
        self.result_windows = {"tm": None, "ce": None, "plot": None} # 移除了 "ct"
        self.saved_real_coords = [None] * 3 # 保存的实际坐标，用于控制点表
        self.selected_control_point_row = -1 # 控制点表中选中的行
        self._create_actions() # 创建动作
        self._create_menu_bar() # 创建菜单栏
        self._create_status_bar() # 创建状态栏
        self._create_main_widget() # 创建主控件
        # 将窗口移动到屏幕中央
        center = QGuiApplication.primaryScreen().availableGeometry().center()
        self.move(center.x() - self.width() / 2, center.y() - self.height() / 2)
        print("提示：程序没有内置撤销功能。如需修正参数或选择，请重新执行相应步骤。")

    def _create_actions(self):
        """创建应用程序的 QAction。"""
        # --- 为 QAction 添加图标 ---
        # icon_load = QIcon("图标/icons8-48.png")
        # icon_save = QIcon("图标/icons8-32.png")
        # icon_exit = QIcon("图标/icons8-50.png")
        # icon_fit = QIcon("图标/icons8-窗口-48.png")
        # icon_show_tm = QIcon("图标/icons8-缩略图-100.png")
        # icon_show_ce = QIcon("图标/icons8-缩略图-64.png")
        # icon_show_plot = QIcon("图标/icons8-组合图-48.png")

        # 检查图标文件是否存在，如果不存在则不设置图标
        def create_action(icon_rel_path, text, parent, triggered_func):
            """辅助函数，用于创建带图标的 QAction。"""
            icon_path = resource_path(icon_rel_path)  # <--- 使用辅助函数获取绝对路径
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                return QAction(icon, text, parent, triggered=triggered_func)
            else:
                print(f"警告: 图标文件未找到: {icon_path} (原始相对路径: {icon_rel_path})")
                return QAction(text, parent, triggered=triggered_func)

        self.load_action = create_action("图标/icons8-48.png", "&加载图像...", self, self.load_images)
        self.save_results_action = create_action("图标/icons8-32.png", "&保存结果...", self, self.save_results)
        self.exit_action = create_action("图标/icons8-50.png", "&退出", self, self.close)

        self.fit_view_action = create_action("图标/icons8-窗口-48.png", "适应窗口", self, self.fit_view)

        self.show_tm_action = create_action("图标/icons8-缩略图-100.png", "显示匹配结果", self,
                                            lambda: self.show_image_result_window("tm"))
        self.show_ce_action = create_action("图标/icons8-缩略图-64.png", "显示坐标提取结果", self,
                                            lambda: self.show_image_result_window("ce"))
        self.show_plot_action = create_action("图标/icons8-组合图-48.png", "显示多期变化图", self,
                                              self.show_plot_window)
        # --- 图标修改结束 ---

    def _create_menu_bar(self):
        """创建菜单栏。"""
        menu_bar = self.menuBar()

        # --- 为 QMenu 添加图标 ---
        # icon_file_menu = QIcon("图标/icons8-图像文件-50.png")  # 文件菜单图标
        # icon_view_menu = QIcon("图标/icons8-视图-50.png")  # 视图菜单图标

        file_menu = menu_bar.addMenu("&文件")
        icon_file_menu_path = resource_path("图标/icons8-图像文件-50.png")  # <--- 使用辅助函数
        if os.path.exists(icon_file_menu_path):
            icon_file_menu = QIcon(icon_file_menu_path)
            file_menu.setIcon(icon_file_menu)
        # --- 文件菜单图标设置结束 ---

        file_menu.addAction(self.load_action)
        file_menu.addAction(self.save_results_action)
        file_menu.addSeparator() # 添加分隔符
        file_menu.addAction(self.exit_action)

        view_menu = menu_bar.addMenu("&视图")
        icon_view_menu_path = resource_path("图标/icons8-视图-50.png")  # <--- 使用辅助函数
        if os.path.exists(icon_view_menu_path):
            icon_view_menu = QIcon(icon_view_menu_path)
            view_menu.setIcon(icon_view_menu)
        # --- 视图菜单图标设置结束 ---

        view_menu.addAction(self.fit_view_action)
        view_menu.addSeparator()
        view_menu.addAction(self.show_tm_action)
        view_menu.addAction(self.show_ce_action)
        view_menu.addSeparator()
        view_menu.addAction(self.show_plot_action)

    def _create_status_bar(self):
        """创建状态栏。"""
        # ... (状态栏保持不变) ...
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪") # 设置初始状态信息
        self.progress_bar = QProgressBar() # 创建进度条
        self.progress_bar.setMaximumWidth(200) # 设置最大宽度
        self.progress_bar.setMaximumHeight(15) # 设置最大高度
        self.progress_bar.setTextVisible(False) # 不显示文本
        self.progress_bar.setRange(0, 0) # 设置为不确定模式
        self.progress_bar.hide() # 初始隐藏
        self.status_bar.addPermanentWidget(self.progress_bar) # 将进度条添加到状态栏

    def _create_main_widget(self):
        """创建主窗口的中心控件。"""
        # ... (控件创建，使用更新后的默认值) ...
        main_widget = QWidget() # 创建主控件
        main_layout = QHBoxLayout(main_widget) # 创建水平布局

        # 左侧面板
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel) # 设置边框样式
        left_panel.setMaximumWidth(400) # 设置最大宽度
        left_layout = QVBoxLayout(left_panel) # 创建垂直布局

        self.image_list_widget = QListWidget() # 图像列表控件
        self.image_list_widget.currentItemChanged.connect(self.on_image_selection_changed) # 连接当前项改变信号
        self.image_list_widget.setContextMenuPolicy(Qt.CustomContextMenu) # 设置上下文菜单策略
        self.image_list_widget.customContextMenuRequested.connect(self.show_list_context_menu) # 连接自定义上下文菜单请求信号
        left_layout.addWidget(QLabel("加载的图像列表"))
        left_layout.addWidget(self.image_list_widget)

        # 模板匹配组
        tm_group = QGroupBox("1. 模板匹配")
        tm_layout = QVBoxLayout(tm_group)
        tm_thresh_layout = QHBoxLayout()
        tm_thresh_layout.addWidget(QLabel("匹配阈值:"))
        self.tm_thresh_spinbox = QDoubleSpinBox() # 匹配阈值微调框
        self.tm_thresh_spinbox.setRange(0.1, 1.0)
        self.tm_thresh_spinbox.setSingleStep(0.05)
        self.tm_thresh_spinbox.setDecimals(2)
        self.tm_thresh_spinbox.setValue(0.5) # 默认值
        tm_thresh_layout.addWidget(self.tm_thresh_spinbox)
        tm_layout.addLayout(tm_thresh_layout)
        self.select_template_button = QPushButton("框选模板区域") # 选择模板区域按钮
        self.select_template_button.clicked.connect(self.activate_template_selection)
        self.run_tm_button = QPushButton("执行模板匹配") # 执行模板匹配按钮
        self.run_tm_button.clicked.connect(self.run_template_matching)
        self.tm_result_label = QLabel("匹配区域数量: N/A") # 模板匹配结果标签
        tm_layout.addWidget(self.select_template_button)
        tm_layout.addWidget(self.run_tm_button)
        tm_layout.addWidget(self.tm_result_label)
        left_layout.addWidget(tm_group)

        # 坐标提取组
        ce_group = QGroupBox("2. 坐标提取")
        ce_layout = QVBoxLayout(ce_group)
        dilate_layout = QHBoxLayout()
        dilate_layout.addWidget(QLabel("膨胀 Kernel 大小 :"))
        self.dilate_kernel_spinbox = QSpinBox() # 膨胀核大小微调框
        self.dilate_kernel_spinbox.setRange(1, 15)
        self.dilate_kernel_spinbox.setSingleStep(2)
        self.dilate_kernel_spinbox.setValue(2) # 默认值 2
        dilate_layout.addWidget(self.dilate_kernel_spinbox)
        ce_layout.addLayout(dilate_layout)
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("最小轮廓面积:"))
        self.area_thresh_spinbox = QSpinBox() # 最小轮廓面积微调框
        self.area_thresh_spinbox.setRange(10, 10000)
        self.area_thresh_spinbox.setValue(100) # 默认值 100
        area_layout.addWidget(self.area_thresh_spinbox)
        ce_layout.addLayout(area_layout)
        circ_layout = QHBoxLayout()
        circ_layout.addWidget(QLabel("最小相似度:")) # 这里应该是圆度或形状相似度
        self.circ_thresh_spinbox = QDoubleSpinBox() # 最小圆度/相似度微调框
        self.circ_thresh_spinbox.setRange(0.1, 1.0)
        self.circ_thresh_spinbox.setSingleStep(0.05)
        self.circ_thresh_spinbox.setDecimals(2)
        self.circ_thresh_spinbox.setValue(0.6) # 默认值 0.6
        circ_layout.addWidget(self.circ_thresh_spinbox)
        ce_layout.addLayout(circ_layout)
        self.run_ce_button = QPushButton("执行坐标提取") # 执行坐标提取按钮
        self.run_ce_button.clicked.connect(self.run_circle_extraction)
        self.ce_result_label = QLabel("提取中心数量: N/A") # 坐标提取结果标签
        ce_layout.addWidget(self.run_ce_button)
        ce_layout.addWidget(self.ce_result_label)
        left_layout.addWidget(ce_group)

        # 坐标转换组
        ct_group = QGroupBox("3. 坐标转换")
        ct_layout = QVBoxLayout(ct_group)
        self.control_point_table = QTableWidget() # 控制点表格
        self.control_point_table.setColumnCount(4)
        self.control_point_table.setHorizontalHeaderLabels(["点号", "像素 X", "像素 Y", "实际坐标 (X,Y)"])
        self.control_point_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) # 列宽自动拉伸
        self.control_point_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents) # 第四列根据内容调整
        self.control_point_table.setRowCount(3) # 默认3行控制点
        self.control_point_table.cellClicked.connect(self.on_table_cell_clicked) # 连接单元格点击信号
        self.control_point_table.setSelectionBehavior(QAbstractItemView.SelectRows) # 设置选择行为为选择整行
        self.control_point_table.setSelectionMode(QAbstractItemView.SingleSelection) # 设置选择模式为单选
        self._reset_control_point_table() # 初始化控制点表格
        self.select_pixel_button = QPushButton("点击选择像素坐标") # 选择像素坐标按钮
        self.select_pixel_button.setCheckable(True) # 设置为可选中状态
        self.select_pixel_button.clicked.connect(self.activate_point_selection)
        self.run_ct_button = QPushButton("执行坐标转换 (当前期)") # 执行坐标转换按钮
        self.run_ct_button.clicked.connect(self.run_coordinate_transformation)
        self.ct_result_label = QLabel("状态: 未转换") # 坐标转换结果标签
        ct_layout.addWidget(self.control_point_table)
        ct_layout.addWidget(self.select_pixel_button)
        ct_layout.addWidget(self.run_ct_button)
        ct_layout.addWidget(self.ct_result_label)
        left_layout.addWidget(ct_group)
        left_layout.addStretch() # 添加伸缩因子
        main_layout.addWidget(left_panel)

        # 中心面板
        center_panel = QFrame()
        center_panel.setFrameShape(QFrame.StyledPanel)
        center_layout = QVBoxLayout(center_panel)
        self.image_scene = QGraphicsScene() # 图像场景
        self.image_view = InteractiveGraphicsView(self.image_scene) # 图像视图
        self.image_view.regionSelected.connect(self.handle_region_selected) # 连接区域选择信号
        self.image_view.pointClicked.connect(self.handle_point_clicked) # 连接点点击信号
        center_layout.addWidget(QLabel("图像操作区"))
        center_layout.addWidget(self.image_view)
        main_layout.addWidget(center_panel, 1) # 设置伸展因子为1，使其占据更多空间
        self.setCentralWidget(main_widget) # 设置中心控件


    def _reset_control_point_table(self, pixel_coords_to_load=None):
        """重置控制点表格内容。"""
        # ... (实现保持不变) ...
        self.control_point_table.clearContents() # 清空表格内容
        num_rows = self.control_point_table.rowCount()
        if pixel_coords_to_load is None:
            pixel_coords_to_load = [None] * num_rows
        elif len(pixel_coords_to_load) != num_rows:
            # 如果加载的坐标数量与行数不匹配，进行调整
            coords_temp = [None] * num_rows
            for i in range(min(len(pixel_coords_to_load), num_rows)):
                coords_temp[i] = pixel_coords_to_load[i] # 使用 insert 会导致问题，直接赋值
            pixel_coords_to_load = coords_temp

        for i in range(num_rows):
            self.control_point_table.setItem(i, 0, QTableWidgetItem(f"P{i+1}")) # 设置点号
            pixel_coord = pixel_coords_to_load[i]
            px_text = f"{pixel_coord[0]:.2f}" if pixel_coord else "点击选择"
            py_text = f"{pixel_coord[1]:.2f}" if pixel_coord else "点击选择"
            self.control_point_table.setItem(i, 1, QTableWidgetItem(px_text)) # 设置像素X
            self.control_point_table.setItem(i, 2, QTableWidgetItem(py_text)) # 设置像素Y
            saved_coord = self.saved_real_coords[i] if i < len(self.saved_real_coords) else None
            real_coord_text = f"{saved_coord[0]},{saved_coord[1]}" if saved_coord else "输入 X,Y"
            self.control_point_table.setItem(i, 3, QTableWidgetItem(real_coord_text)) # 设置实际坐标
            # 设置单元格标志，使其可选但不可编辑（实际坐标除外）
            self.control_point_table.item(i, 0).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.control_point_table.item(i, 1).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.control_point_table.item(i, 2).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.control_point_table.item(i, 3).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        self.control_point_table.clearSelection() # 清除选择
        self.selected_control_point_row = -1 # 重置选中的行

    # --- 槽函数方法 ---
    def load_images(self):
        """加载图像文件。"""
        # *** 追加图像 ***
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)") # 设置文件过滤器
        file_dialog.setFileMode(QFileDialog.ExistingFiles) # 设置文件模式为选择多个已存在的文件
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            added_count = 0
            newly_added_items = []
            current_paths = list(self.image_files.values()) # 获取当前已加载的图像路径列表
            for file_path in selected_files:
                if file_path and file_path not in current_paths: # 避免重复加载
                    item_text = os.path.basename(file_path) # 获取文件名作为显示文本
                    display_text = item_text
                    counter = 1
                    # 如果文件名已存在，则添加后缀以区分
                    while self.image_list_widget.findItems(display_text, Qt.MatchExactly):
                        display_text = f"{item_text} ({counter})"
                        counter += 1
                    list_item = QListWidgetItem(display_text)
                    self.image_list_widget.addItem(list_item) # 添加到列表控件
                    self.image_files[display_text] = file_path # 存储文件路径
                    self.processing_results[file_path] = {} # 为新图像初始化处理结果字典
                    newly_added_items.append(list_item)
                    added_count += 1
            if added_count > 0:
                self.status_bar.showMessage(f"添加了 {added_count} 张新图像")
                if newly_added_items:
                    self.image_list_widget.setCurrentItem(newly_added_items[0]) # 选中第一个新添加的图像
            else:
                self.status_bar.showMessage("未添加新图像（可能已存在）")


    def on_image_selection_changed(self, current_item, previous_item):
        """处理图像列表选择变化的槽函数。"""
        if current_item:
            # *** 窗口在选择更改时不会关闭 ***
            selected_text = current_item.text()
            new_image_path = self.image_files.get(selected_text)
            if new_image_path and new_image_path != self.current_image_path:
                self.current_image_path = new_image_path
                reader = QImageReader(self.current_image_path) # 使用 QImageReader 以支持自动变换
                reader.setAutoTransform(True)
                image = reader.read()
                if image.isNull():
                    self.status_bar.showMessage(f"错误: 无法加载图像 {self.current_image_path} - {reader.errorString()}")
                    self.image_scene.clear()
                    self.image_view.resetTransform()
                else:
                    pixmap = QPixmap.fromImage(image)
                    self.image_scene.clear()
                    self.image_scene.addPixmap(pixmap)
                    self.image_scene.setSceneRect(QRectF(pixmap.rect()))
                    self.fit_view() # 适应视图
                    self.status_bar.showMessage(f"当前图像: {selected_text} (原始尺寸: {pixmap.width()}x{pixmap.height()})")
                    # 重置选择模式
                    self.select_pixel_button.setChecked(False)
                    self.image_view.set_selecting_point_mode(False)
                    self.image_view.set_selecting_region_mode(False)
                    # 加载或初始化当前图像的控制点
                    if self.current_image_path not in self.processing_results:
                        self.processing_results[self.current_image_path] = {}
                    current_results = self.processing_results[self.current_image_path]
                    num_rows = self.control_point_table.rowCount()
                    pixel_coords = current_results.get('control_points_pixel', [None] * num_rows)
                    if len(pixel_coords) != num_rows: # 确保长度一致
                        pixel_coords = [None] * num_rows
                    current_results['control_points_pixel'] = pixel_coords # 存储或更新像素坐标
                    self._reset_control_point_table(pixel_coords_to_load=pixel_coords) # 重置表格并加载数据
                    self.update_result_displays() # 更新结果显示标签


    def close_all_result_windows(self, exclude_plot=False):
        """关闭所有结果显示窗口。"""
        window_refs = list(self.result_windows.keys()) # 获取所有窗口引用
        if exclude_plot and 'plot' in window_refs:
            window_refs.remove('plot') # 如果排除绘图窗口，则移除
        for window_ref in window_refs:
            window = self.result_windows.get(window_ref)
            if window:
                try:
                    window.windowClosed.disconnect() # 断开信号连接，避免重复处理
                except RuntimeError: # 如果信号未连接，会抛出 RuntimeError
                    pass
                window.close() # 关闭窗口
                self.result_windows[window_ref] = None # 将引用置为 None

    def fit_view(self):
        """适应视图以显示整个图像。"""
        self.image_view.fit_view_to_scene()

    def activate_template_selection(self):
        """激活模板选择模式。"""
        if self.select_pixel_button.isChecked(): # 如果点选按钮被选中，则取消
            self.select_pixel_button.setChecked(False)
            self.image_view.set_selecting_point_mode(False)
        if self.current_image_path:
            self.status_bar.showMessage("请在图像上拖拽鼠标框选模板区域...")
            self.image_view.set_selecting_region_mode(True) # 设置为区域选择模式
        else:
            self.status_bar.showMessage("请先加载并选择一张图像")

    def handle_region_selected(self, rect):
        """处理模板区域选择完成的槽函数。"""
        self.image_view.set_selecting_region_mode(False) # 关闭区域选择模式
        if self.current_image_path:
            if rect.width() < 5 or rect.height() < 5: # 检查模板大小
                self.status_bar.showMessage("模板区域太小，请重新选择")
                return
            self.status_bar.showMessage(f"模板区域已选择: {rect.x():.1f},{rect.y():.1f} W:{rect.width():.1f} H:{rect.height():.1f}")
            if self.current_image_path not in self.processing_results:
                self.processing_results[self.current_image_path] = {}
            # 存储模板矩形信息
            self.processing_results[self.current_image_path]['template_rect'] = rect
            self.processing_results[self.current_image_path]['template_w'] = int(rect.width())
            self.processing_results[self.current_image_path]['template_h'] = int(rect.height())

    @Slot(int, int)
    def on_table_cell_clicked(self, row, column):
        """处理控制点表格单元格点击的槽函数。"""
        if column == 0: # 如果点击的是第一列（点号列）
            self.selected_control_point_row = row # 记录选中的行
            self.status_bar.showMessage(f"已选中控制点 P{row+1}，请激活像素选择并在图像上点击")
        elif self.selected_control_point_row != -1: # 如果点击其他列且之前有选中行，则清除选择
            self.control_point_table.clearSelection()
            self.selected_control_point_row = -1

    def activate_point_selection(self):
        """激活像素点选择模式。"""
        is_checked = self.select_pixel_button.isChecked()
        if is_checked and self.image_view.selecting_region: # 如果区域选择模式激活，则关闭
            self.image_view.set_selecting_region_mode(False)
        if is_checked and self.current_image_path:
            prompt = f"请在图像上点击以选择 P{self.selected_control_point_row + 1} 的像素坐标" \
                if self.selected_control_point_row >= 0 else "请在图像上点击以顺序选择像素坐标"
            self.status_bar.showMessage(prompt)
            self.image_view.set_selecting_point_mode(True) # 设置为点选择模式
        elif not is_checked:
            self.image_view.set_selecting_point_mode(False) # 关闭点选择模式
            self.status_bar.showMessage("像素点选择模式已关闭")
        elif not self.current_image_path:
            self.status_bar.showMessage("请先加载并选择一张图像")
            self.select_pixel_button.setChecked(False) # 取消按钮选中状态

    def handle_point_clicked(self, point):
        """处理图像上点点击的槽函数（用于控制点选择）。"""
        if not self.current_image_path:
            return
        if self.current_image_path not in self.processing_results:
            self.processing_results[self.current_image_path] = {}

        num_rows = self.control_point_table.rowCount()
        # 确保 'control_points_pixel' 存在且长度正确
        if 'control_points_pixel' not in self.processing_results[self.current_image_path] or \
           len(self.processing_results[self.current_image_path]['control_points_pixel']) != num_rows:
            self.processing_results[self.current_image_path]['control_points_pixel'] = [None] * num_rows

        row_to_update = -1
        if self.selected_control_point_row >= 0: # 如果表格中有选中的行，则更新该行
            row_to_update = self.selected_control_point_row
            self.status_bar.showMessage(f"已更新 P{row_to_update + 1} 像素坐标")
            self.select_pixel_button.setChecked(False) # 关闭点选模式
            self.image_view.set_selecting_point_mode(False)
            self.control_point_table.clearSelection() # 清除表格选择
            self.selected_control_point_row = -1 # 重置选中行
        else: # 否则，顺序查找第一个未设置的控制点行
            pixel_coords_stored = self.processing_results[self.current_image_path]['control_points_pixel']
            for row in range(num_rows):
                if pixel_coords_stored[row] is None:
                    row_to_update = row
                    break
            if row_to_update != -1:
                self.status_bar.showMessage(f"已自动选择 P{row_to_update + 1} 像素坐标")
            else: # 所有行都已填充
                self.status_bar.showMessage("所有控制点像素坐标均已选择，请先选中要覆盖的点号")
                self.select_pixel_button.setChecked(False)
                self.image_view.set_selecting_point_mode(False)
                return

        if row_to_update != -1:
            # 更新表格和内部存储的像素坐标
            if self.control_point_table.item(row_to_update, 1) is None:
                self.control_point_table.setItem(row_to_update, 1, QTableWidgetItem())
            if self.control_point_table.item(row_to_update, 2) is None:
                self.control_point_table.setItem(row_to_update, 2, QTableWidgetItem())
            self.control_point_table.item(row_to_update, 1).setText(f"{point.x():.2f}")
            self.control_point_table.item(row_to_update, 2).setText(f"{point.y():.2f}")
            self.processing_results[self.current_image_path]['control_points_pixel'][row_to_update] = [point.x(), point.y()]

        # 检查是否所有控制点都已选择
        filled_rows = sum(1 for coord in self.processing_results[self.current_image_path]['control_points_pixel'] if coord is not None)
        if filled_rows >= num_rows and self.selected_control_point_row == -1: # 且不是通过选中行更新的
            self.select_pixel_button.setChecked(False)
            self.image_view.set_selecting_point_mode(False)
            self.status_bar.showMessage("已选择足够数量的控制点像素坐标")

    # --- NMS 辅助函数 ---
    def apply_nms_manual(self, res, threshold, nms_distance, template_w, template_h):
        """
        手动实现非极大值抑制 (NMS)。
        """
        loc = np.where(res >= threshold) # 找到所有大于阈值的匹配位置
        points_scores = list(zip(loc[1], loc[0], res[loc[0], loc[1]])) # (x, y, score)
        if not points_scores:
            return []
        points_scores.sort(key=lambda x: x[2], reverse=True) # 按分数降序排序
        kept_points = []
        while points_scores:
            best_x, best_y, best_score = points_scores.pop(0) # 取出分数最高的点
            kept_points.append((best_x, best_y))
            remaining_points = []
            # 移除与最佳点重叠过多的点
            for x, y, score in points_scores:
                if abs(x - best_x) > nms_distance or abs(y - best_y) > nms_distance:
                    remaining_points.append((x, y, score))
            points_scores = remaining_points
        return kept_points

    # --- 任务执行方法 ---
    def run_template_matching(self):
        """执行模板匹配任务。"""
        if not self.current_image_path:
            self.status_bar.showMessage("错误: 未选择图像")
            return
        current_results = self.processing_results.get(self.current_image_path, {})
        if 'template_rect' not in current_results or not isinstance(current_results['template_rect'], QRectF):
            self.status_bar.showMessage("错误: 未选择有效的模板区域")
            return

        self.status_bar.showMessage("正在执行模板匹配 (含NMS)...")
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0) # 不确定模式
        template_rect = current_results['template_rect']
        threshold = self.tm_thresh_spinbox.value()

        def tm_task(image_path, template_rect_data, tm_threshold, signals, nms_function):
            """模板匹配的后台任务函数。"""
            # ... (tm_task 计算逻辑不变, 返回 pixmap) ...
            signals.status_update.emit("模板匹配线程开始 (实现NMS)...")
            import time
            time.sleep(0.1) # 模拟耗时
            try:
                img_color = cv2.imread(image_path)
                if img_color is None:
                    raise ValueError("无法加载图像")
                gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) # 转为灰度图
                # 提取模板
                x, y, w, h = map(int, [template_rect_data.x(), template_rect_data.y(),
                                       template_rect_data.width(), template_rect_data.height()])
                x = max(0, x); y = max(0, y) # 确保坐标不越界
                w = min(w, gray_img.shape[1] - x); h = min(h, gray_img.shape[0] - y) # 确保宽高不越界
                if w <= 0 or h <= 0:
                    raise ValueError("模板尺寸无效")
                template = gray_img[y:y+h, x:x+w]
                # 执行模板匹配
                res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
                # 应用 NMS
                nms_distance = min(w, h) // 2 # NMS 距离阈值，设为模板短边的一半
                final_points = nms_function(res, tm_threshold, nms_distance, w, h)
                num_matches = len(final_points)
                # 在结果图像上绘制矩形框
                result_img = img_color.copy()
                for (pt_x, pt_y) in final_points:
                    cv2.rectangle(result_img, (pt_x, pt_y), (pt_x + w, pt_y + h), (255, 0, 0), 1) # 蓝色矩形
                # 将 OpenCV 图像转换为 QPixmap
                height_res, width_res, channel_res = result_img.shape
                bytes_per_line = 3 * width_res
                q_img = QImage(result_img.data, width_res, height_res, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                result_pixmap = QPixmap.fromImage(q_img)
                # 发出结果信号
                signals.new_image_result.emit("tm", result_pixmap, image_path)
                signals.status_update.emit(f"模板匹配完成 (NMS后): {num_matches} 个匹配区域 (阈值={tm_threshold:.2f})")
                return {"matched_regions_count": num_matches, "final_points": final_points,
                        "template_w": w, "template_h": h, "tm_pixmap": result_pixmap}
            except Exception as e:
                import traceback
                print(f"模板匹配任务错误: {e}\n{traceback.format_exc()}")
                signals.status_update.emit(f"模板匹配错误: {e}")
                return {"error": str(e)}

        # 创建并启动工作线程
        worker = Worker(tm_task, self.current_image_path, template_rect, threshold, nms_function=self.apply_nms_manual)
        thread = QThread()
        self.threadpool.append((thread, worker)) # 添加到线程池管理
        worker.moveToThread(thread)
        worker.signals.result.connect(self.handle_tm_result) # 连接结果处理槽函数
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.signals.status_update.connect(self.status_bar.showMessage) # 连接状态更新槽函数
        worker.signals.error.connect(lambda x: self.status_bar.showMessage(f"线程错误: {x}"))
        worker.signals.new_image_result.connect(self.update_result_window) # 连接新图像结果更新槽函数
        worker.signals.finished.connect(self.trigger_ce_after_tm) # 模板匹配完成后触发坐标提取
        thread.started.connect(worker.run)
        thread.start()
        self.run_tm_button.setEnabled(False) # 禁用按钮，防止重复点击
        worker.signals.finished.connect(lambda: self.run_tm_button.setEnabled(True)) # 任务完成后恢复按钮
        worker.signals.finished.connect(self.hide_progress) # 任务完成后隐藏进度条

    def handle_tm_result(self, result_data):
        """处理模板匹配结果的槽函数。"""
        # *** 存储 pixmap ***
        if self.current_image_path:
            if "error" in result_data:
                self.tm_result_label.setText("匹配区域数量: 错误")
                self.status_bar.showMessage(f"模板匹配失败: {result_data['error']}")
            elif "matched_regions_count" in result_data:
                count = result_data["matched_regions_count"]
                self.tm_result_label.setText(f"匹配区域数量: {count}")
                if self.current_image_path not in self.processing_results:
                    self.processing_results[self.current_image_path] = {}
                current_results = self.processing_results[self.current_image_path]
                # 存储模板匹配结果
                current_results['tm_count'] = count
                current_results['final_points'] = result_data.get("final_points", [])
                current_results['template_w'] = result_data.get("template_w", 0)
                current_results['template_h'] = result_data.get("template_h", 0)
                if "tm_pixmap" in result_data:
                    current_results['tm_pixmap'] = result_data["tm_pixmap"] # 存储 pixmap
                self.status_bar.showMessage(f"模板匹配完成，结果已内部保存: {self.current_image_path}")

    @Slot()
    def trigger_ce_after_tm(self):
        """在模板匹配成功后自动触发圆心提取。"""
        # *** 检查 tm_count > 0 ***
        if self.current_image_path and self.current_image_path in self.processing_results:
            current_results = self.processing_results[self.current_image_path]
            if current_results.get('tm_count', 0) > 0 and 'final_points' in current_results:
                print("模板匹配找到区域，自动触发圆心提取...")
                self.status_bar.showMessage("模板匹配找到区域，自动执行圆心提取...")
                self.run_circle_extraction() # 执行圆心提取
            elif current_results.get('tm_count', -1) == 0:
                print("模板匹配完成但找到 0 个区域，跳过自动圆心提取。")
                self.status_bar.showMessage("模板匹配未找到区域，请手动执行圆心提取或调整参数。")
            else:
                print("模板匹配结果无效/失败，不自动触发圆心提取。")
                self.status_bar.showMessage("模板匹配结果无效/失败，请手动执行圆心提取。")
        else:
            print("模板匹配完成但当前图像路径无效，不自动触发圆心提取。")

    def run_circle_extraction(self):
        """执行圆心提取任务。"""
        if not self.current_image_path:
            self.status_bar.showMessage("错误: 未选择图像")
            return
        current_results = self.processing_results.get(self.current_image_path, {})
        final_points = current_results.get('final_points') # 获取模板匹配找到的点
        template_w = current_results.get('template_w') # 获取模板宽度
        template_h = current_results.get('template_h') # 获取模板高度

        if final_points is None or template_w is None or template_h is None:
            self.status_bar.showMessage("错误: 请先成功执行模板匹配")
            return
        # 如果模板匹配没有找到点，也在此处明确处理
        if not final_points:
            self.status_bar.showMessage("模板匹配未找到任何区域，无法提取中心")
            self.ce_result_label.setText("提取中心数量: 0")
            if 'pixel_coords' in current_results: del current_results['pixel_coords']  # 清除之前的坐标提取结果
            if 'ce_count' in current_results: del current_results['ce_count']
            empty_pix = QPixmap(100, 100)
            empty_pix.fill(Qt.transparent)  # 简单的空 pixmap
            self.update_result_window("ce", empty_pix, self.current_image_path) # 更新结果窗口
            return

        self.status_bar.showMessage("正在执行圆心提取 (基于模板匹配区域)...")
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        kernel_size = self.dilate_kernel_spinbox.value() # 获取膨胀核大小
        area_thresh = self.area_thresh_spinbox.value() # 获取最小轮廓面积阈值
        circ_thresh = self.circ_thresh_spinbox.value() # 获取最小圆度阈值

        # --- CE 任务定义 (基于区域) ---
        def ce_task(image_path, tm_points, tm_w, tm_h, k_size, a_thresh, c_thresh, signals):
            """圆心提取的后台任务函数（基于模板匹配区域）。"""
            signals.status_update.emit("圆心提取线程开始 (区域处理)...")
            import time
            time.sleep(0.1) # 模拟耗时
            try:
                img_color = cv2.imread(image_path)
                if img_color is None:
                    raise ValueError("无法加载图像")
                img_h_full, img_w_full = img_color.shape[:2] # 获取完整图像尺寸

                # 1. 对完整图像进行一次预处理
                gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                signals.status_update.emit("应用双边滤波 (全图)...")
                bilateral_filtered = cv2.bilateralFilter(gray_img, d=9, sigmaColor=65, sigmaSpace=65)
                signals.status_update.emit("应用Otsu阈值 (全图)...")
                ret, otsu_thresh = cv2.threshold(bilateral_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                signals.status_update.emit("应用Canny边缘 (全图)...")
                edges = cv2.Canny(otsu_thresh, 30, 70)
                signals.status_update.emit("应用形态学膨胀 (全图)...")
                kernel = np.ones((k_size, k_size), np.uint8)
                dilation = cv2.dilate(edges, kernel, iterations=1)

                # 2. 处理每个模板匹配区域
                signals.status_update.emit(f"在 {len(tm_points)} 个区域内查找圆心...")
                all_centroids_full_coords = [] # 存储所有找到的圆心（完整图像坐标）
                min_area = float(a_thresh)
                min_circularity = float(c_thresh)

                for i, pt in enumerate(tm_points):
                    pt_x, pt_y = int(pt[0]), int(pt[1])
                    # 定义模板匹配区域的边界
                    y1, y2 = max(0, pt_y), min(img_h_full, pt_y + tm_h)
                    x1, x2 = max(0, pt_x), min(img_w_full, pt_x + tm_w)
                    if y1 >= y2 or x1 >= x2: # 无效区域
                        continue

                    dilation_region = dilation[y1:y2, x1:x2] # 提取该区域的膨胀结果
                    # 在该区域内查找轮廓
                    contours_region, _ = cv2.findContours(dilation_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours_region:
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter == 0:
                            continue
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        # 根据面积和圆度筛选轮廓
                        if area > min_area and circularity > min_circularity:
                            M = cv2.moments(contour) # 计算轮廓矩
                            if M["m00"] != 0: # 避免除以零
                                # 计算轮廓中心点（相对于区域）
                                center_x_rel = int(M["m10"] / M["m00"])
                                center_y_rel = int(M["m01"] / M["m00"])
                                # 转换为完整图像坐标
                                center_x_full = x1 + center_x_rel
                                center_y_full = y1 + center_y_rel
                                all_centroids_full_coords.append([center_x_full, center_y_full])

                # 3. 聚合和过滤所有圆心 (欧氏距离去重)
                signals.status_update.emit(f"聚合 {len(all_centroids_full_coords)} 个候选圆心并去重...")
                unique_centroids_array = np.array([])
                if all_centroids_full_coords:
                    centroids_array = np.array(all_centroids_full_coords)
                    min_dist_threshold = 5.0 # 最小距离阈值，小于此距离的点被认为是同一个
                    to_keep = np.ones(len(centroids_array), dtype=bool) # 标记哪些点需要保留
                    if len(centroids_array) > 1:  # 避免 cdist 处理单个点
                        for i in range(len(centroids_array)):
                            if to_keep[i]:
                                # 将点 i 与点 j > i 且仍标记为保留的点进行比较
                                compare_indices = np.where(to_keep[i + 1:])[0] + (i + 1) # 获取要比较的点的原始索引
                                if compare_indices.size > 0:
                                    distances = cdist(centroids_array[i:i + 1], centroids_array[compare_indices]) # 计算距离
                                    close_relative_indices = np.where(distances < min_dist_threshold)[1]
                                    # 使用 compare_indices 中的原始索引标记要移除的点
                                    to_keep[compare_indices[close_relative_indices]] = False
                    unique_centroids_array = centroids_array[to_keep] # 保留未被标记为移除的点

                num_centers = len(unique_centroids_array)
                signals.status_update.emit(f"去重后最终圆心数量: {num_centers}")

                # 4. 排序最终坐标
                signals.status_update.emit("排序最终坐标...")
                sorted_coords_array = sort_grid_points(unique_centroids_array) if num_centers > 0 else np.array([])

                # 5. 创建结果图像
                result_img = img_color.copy()
                if num_centers > 0:
                    for i in range(num_centers):
                        center = tuple(map(int, sorted_coords_array[i]))
                        cv2.circle(result_img, center, 5, (0, 255, 0), -1) # 绘制绿色圆点
                # 转换为 QPixmap
                height_res, width_res, channel_res = result_img.shape
                bytes_per_line = 3 * width_res
                q_img = QImage(result_img.data, width_res, height_res, bytes_per_line,
                               QImage.Format_RGB888).rgbSwapped()
                result_pixmap = QPixmap.fromImage(q_img)
                # 发出结果信号
                signals.new_image_result.emit("ce", result_pixmap, image_path)
                signals.status_update.emit(f"圆心提取完成 (区域处理): {num_centers} 个圆心")
                return {"extracted_centers_count": num_centers, "sorted_coords": sorted_coords_array}
            except Exception as e:
                import traceback
                print(f"圆心提取任务错误: {e}\n{traceback.format_exc()}")
                signals.status_update.emit(f"圆心提取错误: {e}")
                return {"error": str(e)}

        # 创建并启动工作线程
        worker = Worker(ce_task, self.current_image_path, final_points, template_w, template_h, kernel_size,
                        area_thresh, circ_thresh)
        thread = QThread()
        self.threadpool.append((thread, worker))
        worker.moveToThread(thread)
        worker.signals.result.connect(self.handle_ce_result)
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.signals.status_update.connect(self.status_bar.showMessage)
        worker.signals.error.connect(lambda x: self.status_bar.showMessage(f"线程错误: {x}"))
        worker.signals.new_image_result.connect(self.update_result_window) # 连接新图像结果更新槽函数
        thread.started.connect(worker.run)
        thread.start()
        self.run_ce_button.setEnabled(False) # 禁用按钮
        worker.signals.finished.connect(lambda: self.run_ce_button.setEnabled(True)) # 恢复按钮
        worker.signals.finished.connect(self.hide_progress) # 隐藏进度条


    def handle_ce_result(self, result_data):
        """处理圆心提取结果的槽函数。"""
        # *** 修改：存储 pixmap ***
        if self.current_image_path:
            if "error" in result_data:
                self.ce_result_label.setText("提取中心数量: 错误")
                self.status_bar.showMessage(f"圆心提取失败: {result_data['error']}")
            elif "extracted_centers_count" in result_data:
                count = result_data["extracted_centers_count"]
                coords = result_data["sorted_coords"]
                self.ce_result_label.setText(f"提取中心数量: {count}")
                if self.current_image_path not in self.processing_results:
                    self.processing_results[self.current_image_path] = {}
                current_results = self.processing_results[self.current_image_path]
                # 存储圆心提取结果
                current_results['ce_count'] = count
                current_results['pixel_coords'] = coords # 存储排序后的像素坐标
                if "ce_pixmap" in result_data: # 如果任务返回了 pixmap (虽然当前 ce_task 没有直接返回这个键)
                    current_results['ce_pixmap'] = result_data["ce_pixmap"] # 存储 CE pixmap
                self.status_bar.showMessage(f"圆心提取完成，排序坐标已内部保存: {self.current_image_path}")


    # --- CT 可视化任务 ---
    def run_ct_visualization_task(self, image_path, pixel_coords, real_coords):
        """启动线程生成坐标转换可视化图像。"""
        # *** 启动线程生成 CT 可视化 pixmap ***
        if not isinstance(pixel_coords, np.ndarray) or not isinstance(real_coords, np.ndarray) or \
           pixel_coords.size == 0 or real_coords.size == 0:
            print("坐标转换可视化任务：提供的坐标无效。")
            return
        if len(pixel_coords) != len(real_coords):
            print("坐标转换可视化任务：像素坐标和实际坐标数量不匹配。")
            return

        def ct_visualization_task(img_path, px_coords, r_coords, signals):
            """坐标转换可视化的后台任务函数。"""
            signals.status_update.emit("生成坐标转换可视化...")
            import time
            time.sleep(0.1) # 模拟耗时
            try:
                base_pixmap = None
                result_img = None
                # 使用存储的 CE pixmap 作为基础
                base_pixmap = self.processing_results.get(img_path, {}).get('ce_pixmap')
                if base_pixmap and not base_pixmap.isNull():
                    # 将 QPixmap 转换为 OpenCV 图像
                    qimage = base_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
                    width = qimage.width()
                    height = qimage.height()
                    ptr = qimage.bits()
                    ptr.setsize(qimage.sizeInBytes()) # 重要：设置大小
                    arr = np.array(ptr).reshape(height, width, 3) # 将 QImage 数据转换为 numpy 数组
                    result_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) # 转换为 BGR 格式
                else: # 如果没有 CE 结果，则加载原图
                    result_img = cv2.imread(img_path)
                    if result_img is None:
                        raise ValueError("无法加载基础图像 (CE结果或原图)")
                    # 如果使用原图，并且有像素坐标，则绘制绿色圆点
                    if px_coords.size > 0:
                        for i in range(len(px_coords)):
                            center = tuple(map(int, px_coords[i]))
                            cv2.circle(result_img, center, 5, (0, 255, 0), -1) # 绘制绿色圆点

                # 在图像上绘制实际坐标文本
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_color = (0, 0, 255) # 红色文本
                line_type = 1
                text_offset_x = 8
                text_offset_y = -8
                for i in range(len(px_coords)):
                    px_coord = tuple(map(int, px_coords[i]))
                    r_coord = r_coords[i]
                    text = f"({r_coord[0]:.3f},{r_coord[1]:.3f})"
                    text_x = px_coord[0] + text_offset_x
                    text_y = px_coord[1] + text_offset_y
                    cv2.putText(result_img, text, (text_x, text_y), font, font_scale, font_color, line_type, cv2.LINE_AA)
                # 转换为 QPixmap
                height_res, width_res, channel_res = result_img.shape
                bytes_per_line = 3 * width_res
                q_img = QImage(result_img.data, width_res, height_res, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                result_pixmap = QPixmap.fromImage(q_img)
                # *** 发出带 'ct' 任务名称的 new_image_result ***
                signals.new_image_result.emit("ct", result_pixmap, img_path)
                signals.status_update.emit("坐标转换可视化生成完毕")
                return {"ct_pixmap": result_pixmap} # 返回 pixmap 以便存储
            except Exception as e:
                import traceback
                print(f"坐标转换可视化任务错误: {e}\n{traceback.format_exc()}")
                signals.status_update.emit(f"坐标转换可视化错误: {e}")
                return {"error": str(e)}

        # 创建并启动工作线程
        worker = Worker(ct_visualization_task, image_path, pixel_coords, real_coords)
        thread = QThread()
        self.threadpool.append((thread, worker))
        worker.moveToThread(thread)
        # *** 连接结果处理槽函数以存储 pixmap ***
        worker.signals.result.connect(self.handle_ct_viz_result)
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.signals.status_update.connect(self.status_bar.showMessage)
        worker.signals.error.connect(lambda x: self.status_bar.showMessage(f"线程错误: {x}"))
        # *** 连接显示处理槽函数 ***
        worker.signals.new_image_result.connect(self.update_result_window)
        thread.started.connect(worker.run)
        thread.start()


    def run_coordinate_transformation(self):
        """执行坐标转换任务。"""
        if not self.current_image_path:
            self.status_bar.showMessage("错误: 未选择图像")
            return
        current_results = self.processing_results.get(self.current_image_path, {})
        if 'pixel_coords' not in current_results or not isinstance(current_results['pixel_coords'], np.ndarray):
            self.status_bar.showMessage("错误: 未执行圆心提取或无排序坐标结果")
            return

        pixel_points_ctrl = [] # 控制点的像素坐标
        real_points_ctrl = [] # 控制点的实际坐标
        min_points = 3 # 最少需要的控制点数量
        valid_points = 0 # 有效的控制点数量
        user_provided_real_coords = False # 用户是否在本次运行中提供了新的实际坐标
        temp_saved_real_coords = [None] * self.control_point_table.rowCount() # 临时存储本次运行的实际坐标

        # 从表格中读取控制点信息
        for row in range(self.control_point_table.rowCount()):
            try:
                px_item = self.control_point_table.item(row, 1)
                py_item = self.control_point_table.item(row, 2)
                real_item = self.control_point_table.item(row, 3)
                if px_item and py_item and real_item and \
                   px_item.text() != "点击选择" and py_item.text() != "点击选择" and real_item.text():
                    px = float(px_item.text())
                    py = float(py_item.text())
                    real_coord_text = real_item.text().replace(" ", "") # 去除空格
                    if real_coord_text and "输入" not in real_coord_text: # 确保已输入且不是提示文本
                        real_x, real_y = map(float, real_coord_text.split(',')) # 解析实际坐标
                        pixel_points_ctrl.append([px, py])
                        real_points_ctrl.append([real_x, real_y])
                        valid_points += 1
                        temp_saved_real_coords[row] = [real_x, real_y] # 存储本次输入的实际坐标
                        # 检查是否与已保存的默认实际坐标不同
                        saved = self.saved_real_coords[row]
                        if saved is None or saved != [real_x, real_y]:
                            user_provided_real_coords = True
                    else:
                        pass # 实际坐标未输入或为提示文本
                else:
                    pass # 像素坐标未选择
            except Exception as e:
                self.status_bar.showMessage(f"错误: 解析控制点表格第 {row+1} 行失败 - {e}")
                return

        if valid_points < min_points:
            self.status_bar.showMessage(f"错误: 至少需要 {min_points} 组有效的控制点 (当前 {valid_points} 组)")
            return

        self.status_bar.showMessage("正在执行坐标转换...")
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        pixel_coords_to_transform = current_results['pixel_coords'] # 获取要转换的像素坐标（来自圆心提取）

        # --- CT 任务定义 (仅计算) ---
        def ct_task(pixel_coords_data, control_pixel, control_real, signals):
            """坐标转换的后台任务函数（仅计算）。"""
            # ... (ct_task 计算逻辑不变) ...
            signals.status_update.emit("坐标转换线程开始...")
            import time
            time.sleep(0.1) # 模拟耗时
            try:
                np_control_pixel = np.array(control_pixel, dtype=np.float32)
                np_control_real = np.array(control_real, dtype=np.float32)
                np_pixel_coords = np.array(pixel_coords_data, dtype=np.float32)

                if np_pixel_coords.size == 0: # 如果没有要转换的点
                    signals.status_update.emit("无圆心坐标可转换")
                    return {"real_coords": np.array([]), "pixel_coords": np.array([])} # 也返回空的像素坐标

                # 计算仿射变换矩阵
                transform_matrix = cv2.getAffineTransform(np_control_pixel, np_control_real)
                # 转换为齐次坐标并应用变换
                ones = np.ones((np_pixel_coords.shape[0], 1), dtype=np.float32)
                pixel_coords_homogeneous = np.hstack([np_pixel_coords, ones])
                real_coords = np.dot(pixel_coords_homogeneous, transform_matrix.T)
                signals.status_update.emit(f"坐标转换完成: {len(real_coords)} 个点")
                return {"real_coords": real_coords, "pixel_coords": pixel_coords_data} # 返回转换后的实际坐标和原始像素坐标
            except Exception as e:
                import traceback
                print(f"坐标转换任务错误: {e}\n{traceback.format_exc()}")
                signals.status_update.emit(f"坐标转换错误: {e}")
                return {"error": str(e)}

        # --- 启动 CT 线程 ---
        worker = Worker(ct_task, pixel_coords_to_transform, pixel_points_ctrl, real_points_ctrl)
        thread = QThread()
        self.threadpool.append((thread, worker))
        worker.moveToThread(thread)
        # 连接结果处理槽函数，并传递额外参数
        worker.signals.result.connect(
            lambda result_data, provided=user_provided_real_coords, coords=temp_saved_real_coords:
            self.handle_ct_result(result_data, provided, coords)
        )
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.signals.status_update.connect(self.status_bar.showMessage)
        worker.signals.error.connect(lambda x: self.status_bar.showMessage(f"线程错误: {x}"))
        thread.started.connect(worker.run)
        thread.start()
        self.run_ct_button.setEnabled(False) # 禁用按钮
        worker.signals.finished.connect(lambda: self.run_ct_button.setEnabled(True)) # 恢复按钮
        worker.signals.finished.connect(self.hide_progress) # 隐藏进度条


    def handle_ct_result(self, result_data, user_provided_real_coords, current_run_real_coords):
        """处理坐标转换结果的槽函数。"""
        # *** 修改：触发 CT 可视化任务，不触发绘图显示 ***
        if self.current_image_path:
            if "error" in result_data:
                self.ct_result_label.setText("状态: 转换错误")
                self.status_bar.showMessage(f"坐标转换失败: {result_data['error']}")
            elif "real_coords" in result_data:
                real_coords = result_data["real_coords"]
                num_coords = len(real_coords) if isinstance(real_coords, np.ndarray) else 0
                pixel_coords = result_data.get("pixel_coords") # 如果 real_coords 存在，则 pixel_coords 也应该存在（除非为空）
                self.ct_result_label.setText(f"状态: {num_coords} 点已转换")
                if self.current_image_path not in self.processing_results:
                    self.processing_results[self.current_image_path] = {}
                self.processing_results[self.current_image_path]['real_coords'] = real_coords # 存储转换后的实际坐标
                self.status_bar.showMessage(f"坐标转换完成，真实坐标已内部保存: {self.current_image_path}")

                # 如果用户提供了新的实际坐标，则保存为默认值
                if user_provided_real_coords:
                    num_saved = 0
                    for i in range(min(len(current_run_real_coords), len(self.saved_real_coords))):
                        if current_run_real_coords[i] is not None:
                            self.saved_real_coords[i] = current_run_real_coords[i]
                            num_saved += 1
                    if num_saved >= 3: # 至少需要3个点才能保存
                        print(f"已将 {num_saved} 个实际控制坐标保存为默认值。")

                # 立即触发 CT 可视化任务
                if num_coords > 0 and pixel_coords is not None and pixel_coords.size > 0: # 也检查 pixel_coords 的大小
                    self.run_ct_visualization_task(self.current_image_path, pixel_coords, real_coords)
                elif num_coords == 0:
                    self.status_bar.showMessage(f"坐标转换完成，但无有效坐标点，跳过可视化。")
                    self.processing_results[self.current_image_path]['ct_pixmap'] = None # 无可视化 pixmap
                else:
                    self.status_bar.showMessage(f"坐标转换完成，但缺少像素坐标，无法可视化。")

                # *** 触发后台绘图更新 (不显示窗口) ***
                self.update_comparison_plot_threaded() # 保留此行


    def handle_ct_viz_result(self, result_data):
        """处理坐标转换可视化结果的槽函数。"""
        # *** 存储 CT 可视化 pixmap ***
        if self.current_image_path and self.current_image_path in self.processing_results:
            if "ct_pixmap" in result_data:
                self.processing_results[self.current_image_path]['ct_pixmap'] = result_data["ct_pixmap"]
            elif "error" in result_data:
                print(f"生成坐标转换可视化时出错: {result_data['error']}")
                self.processing_results[self.current_image_path]['ct_pixmap'] = None


    def save_results(self):
        """保存处理结果（仅实际坐标到 CSV）。"""
        # *** 修改：仅保存 real_coords 到 CSV ***
        if not self.processing_results:
            self.status_bar.showMessage("没有可保存的结果。")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择保存CSV结果的文件夹")
        if output_dir:
            saved_count = 0
            error_count = 0
            # 使用 list(self.processing_results.keys()) 以避免运行时字典更改
            # 仅处理存在的图像路径，并按文件名排序
            sorted_paths = sorted(
                [p for p in self.processing_results.keys() if isinstance(p, str) and os.path.exists(p)],
                key=lambda p: os.path.basename(p)
            )

            for img_path in sorted_paths:
                data = self.processing_results.get(img_path, {})
                # 如果存在有效的 real_coords，则保存
                if 'real_coords' in data and isinstance(data['real_coords'], np.ndarray) and data['real_coords'].size > 0:
                    try:
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        # 构建 CSV 文件名
                        csv_filename = os.path.join(output_dir, f"{base_name}_real_coords.csv")
                        # 使用 numpy 或 pandas 保存
                        np.savetxt(csv_filename, data['real_coords'], fmt='%.6f', delimiter=',',
                                   header='X_Real,Y_Real', comments='') # comments='' 避免写入 '#' 开头的注释行
                        # 或者使用 pandas:
                        # df_to_save = pd.DataFrame(data['real_coords'], columns=['X_Real', 'Y_Real'])
                        # df_to_save.to_csv(csv_filename, index=False, float_format='%.6f')
                        saved_count += 1
                    except Exception as e:
                        error_count += 1
                        print(f"保存文件 {csv_filename} 时出错: {e}")
            if saved_count > 0:
                self.status_bar.showMessage(
                    f"成功保存 {saved_count} 个CSV文件到: {output_dir}" +
                    (f" ({error_count} 个错误)" if error_count else "")
                )
            elif error_count > 0:
                self.status_bar.showMessage(f"保存文件时遇到 {error_count} 个错误。")
            else:
                self.status_bar.showMessage("未找到包含有效坐标转换结果的图像进行保存。")


    # --- 结果窗口处理 ---
    @Slot(str, QPixmap, str)
    def update_result_window(self, task_name, pixmap, original_image_path):
        """更新或创建结果显示窗口。"""
        # *** 存储 pixmap 并更新现有窗口（如果可用），或创建新窗口 ***
        # 处理 TM, CE, CT 可视化窗口
        if task_name not in ["tm", "ce", "ct"]: # 忽略未知类型
            return

        window = None
        title = ""
        base_name = os.path.basename(original_image_path or "") # 获取基本文件名
        # window_ref_name = f"{task_name}_window" # 未使用
        pixmap_key = f"{task_name}_pixmap" # 用于在 processing_results 中存储/检索 pixmap 的键

        icon_rel_path = None  # 默认无图标

        # --- 分配图标路径 ---
        if task_name == "tm":
            title = f"模板匹配 - {base_name}"
            icon_rel_path = "图标/icons8-缩略图-100.png"  # TM 结果窗口图标
        elif task_name == "ce":
            title = f"圆心提取 - {base_name}"
            icon_rel_path = "图标/icons8-缩略图-64.png"  # CE 结果窗口图标
        elif task_name == "ct":
            title = f"坐标转换结果 - {base_name}"
            icon_rel_path = "图标/icons8-组合图-48.png" # 如果有 CT 图标
        else:
            return  # 未知类型，不应发生

        # 为此图像路径存储 pixmap
        if original_image_path not in self.processing_results:
            self.processing_results[original_image_path] = {}
        self.processing_results[original_image_path][pixmap_key] = pixmap

        # 获取现有窗口实例
        window = self.result_windows.get(task_name)

        # 确定标题 (冗余，已在上面设置)
        # if task_name == "tm": title=f"模板匹配 - {base_name}"
        # elif task_name == "ce": title=f"圆心提取 - {base_name}"
        # elif task_name == "ct": title=f"坐标转换结果 - {base_name}"

        if not window:  # 如果窗口不存在或已关闭，创建它
            print(f"为以下任务创建新窗口: {task_name}")
            # --- 创建时传递图标路径 ---
            window = ResultDisplayWindow(task_name, title, icon_rel_path=icon_rel_path)
            self.result_windows[task_name] = window # 存储窗口引用
            window.windowClosed.connect(self.on_result_window_closed) # 连接关闭信号
            # else: # 如果窗口已存在，通常不需要重新设置图标，除非你想更新它
            #     pass

        # 更新窗口内容并显示
        if window:
            window.setWindowTitle(title)  # 更新标题可能需要
            window.set_image(pixmap, original_image_path) # 设置图像和源路径
            window.show()
            window.raise_() # 将窗口置于顶层


    @Slot(QPixmap)
    def update_plot_window(self, pixmap):
        """更新或创建多期对比图显示窗口。"""
        title = "多期监测对比图"
        self.processing_results['plot_pixmap'] = pixmap  # 全局存储绘图结果
        icon_rel_path = "图标/icons8-组合图-48.png"  # Plot 结果窗口图标

        window = self.result_windows.get("plot") # 获取绘图窗口引用
        if not window:
            print("创建新的绘图窗口")
            # --- 创建时传递图标路径 ---
            window = ResultDisplayWindow("plot", title, icon_rel_path=icon_rel_path)
            window.setGeometry(300, 300, 900, 700)  # 保持特定大小
            self.result_windows["plot"] = window # 存储窗口引用
            window.windowClosed.connect(self.on_result_window_closed) # 连接关闭信号
        # else:
        #     pass

        window.setWindowTitle(title)
        window.set_image(pixmap) # 设置图像（绘图结果）
        window.show()
        window.raise_()

    @Slot(str)
    def on_result_window_closed(self, window_type):
        """当结果窗口被用户关闭时调用的槽函数。"""
        # *** 用户关闭窗口时重置引用 ***
        print(f"结果窗口已由用户关闭: {window_type}")
        if window_type in self.result_windows:
            self.result_windows[window_type] = None # 将窗口引用置为 None

    def show_image_result_window(self, window_type):
        """显示当前图像的存储结果，如果窗口不存在则重新创建并传递图标路径。"""
        # *** 显示当前图像的存储结果，重新创建时传递图标路径 ***
        source_path = self.current_image_path # 获取当前选中的图像路径
        if not source_path:
            self.status_bar.showMessage("错误: 未选择图像")
            return
        if source_path not in self.processing_results:
            self.processing_results[source_path] = {}  # 确保字典存在

        window = self.result_windows.get(window_type) # 获取窗口引用
        pixmap = None
        title = ""
        icon_rel_path = None  # <-- 初始化图标相对路径
        base_name = os.path.basename(source_path or "")
        pixmap_key = f"{window_type}_pixmap"  # 例如, tm_pixmap

        # 获取存储的 pixmap
        pixmap = self.processing_results[source_path].get(pixmap_key)

        # --- 根据窗口类型确定标题和图标路径 ---
        if window_type == "tm":
            title = f"模板匹配 - {base_name}"
            icon_rel_path = "图标/icons8-缩略图-100.png"  # <-- TM 图标路径
        elif window_type == "ce":
            title = f"圆心提取 - {base_name}"
            icon_rel_path = "图标/icons8-缩略图-64.png"  # <-- CE 图标路径
        elif window_type == "ct": # 对应坐标转换可视化
            title = f"坐标转换结果 - {base_name}"
            icon_rel_path = "图标/icons8-组合图-48.png" # <-- 如果有 CT 图标路径
        else:
            self.status_bar.showMessage(f"未知窗口类型 '{window_type}'")
            return
        # --- 确定结束 ---

        if pixmap: # 如果有结果图像
            if not window:  # 如果窗口被关闭了，则重新创建
                print(f"为以下任务重新创建窗口: {window_type}")
                # --- 创建时不传递 parent=self ---
                window = ResultDisplayWindow(window_type, title, icon_rel_path=icon_rel_path)  # <-- 移除 parent=self
                self.result_windows[window_type] = window
                window.windowClosed.connect(self.on_result_window_closed)

            # 更新窗口内容并显示/置顶
            window.setWindowTitle(title)  # 确保标题总是最新的
            window.set_image(pixmap, source_path) # 设置图像和源路径
            window.show()
            window.raise_()  # 将窗口带到最前面
        else:
            # 如果没有对应的结果 pixmap
            self.status_bar.showMessage(f"'{title}' 结果尚未生成或未存储，请先运行相应处理步骤。")



    def show_plot_window(self):
        """显示存储的全局绘图结果，如果窗口不存在则重新创建并传递图标路径。"""
        # *** 显示存储的全局绘图结果，重新创建时传递图标路径 ***
        window_type = "plot"
        window = self.result_windows.get(window_type) # 获取绘图窗口引用
        # 从全局结果中获取 plot pixmap
        pixmap = self.processing_results.get('plot_pixmap')
        title = "多期监测对比图"
        icon_rel_path = "图标/icons8-组合图-48.png" # <-- Plot 图标路径

        if pixmap: # 如果有绘图结果
            if not window: # 如果窗口被关闭了，则重新创建
                print(f"为绘图重新创建窗口")
                # --- 创建时不传递 parent=self ---
                window = ResultDisplayWindow(window_type, title, icon_rel_path=icon_rel_path) # <-- 移除 parent=self
                self.result_windows["plot"] = window
                window.windowClosed.connect(self.on_result_window_closed)
                window.setGeometry(300, 300, 900, 700) # 保持特定大小

            # 更新窗口内容并显示/置顶
            window.setWindowTitle(title)
            window.set_image(pixmap) # 绘图窗口不需要 source_path
            window.show()
            window.raise_() # 将窗口带到最前面
        else:
            # 如果没有绘图结果
            self.status_bar.showMessage("多期变化图尚未生成，请先完成至少两期坐标转换。")
            # 可以考虑在这里触发一次绘图尝试：
            # self.update_comparison_plot_threaded()
            # self.status_bar.showMessage("多期变化图尚未生成，正在后台尝试生成...")


    # --- 列表控件的右键菜单 ---
    def show_list_context_menu(self, point):
        """显示图像列表控件的右键上下文菜单。"""
        # *** 显示带图标的列表控件上下文菜单 ***
        item = self.image_list_widget.itemAt(point) # 获取点击位置的项
        # idx = self.image_list_widget.indexAt(point) # idx 当前未使用
        if not item:
            return

        menu = QMenu(self) # 创建菜单
        item_text = item.text()
        image_path = self.image_files.get(item_text) # 获取图像路径

        if not image_path:
            print(f"错误: 未找到项 '{item_text}' 的路径")
            return

        # --- 定义图标路径并获取图标 ---
        icon_remove_rel_path = "图标/icons8-删除图片-48.png"  # <-- 指定移除图标的相对路径
        icon_remove = QIcon()  # 默认空图标
        icon_path_remove_actual = resource_path(icon_remove_rel_path)
        if os.path.exists(icon_path_remove_actual):
            icon_remove = QIcon(icon_path_remove_actual)
        else:
            print(f"警告: 右键菜单移除图标未找到: {icon_path_remove_actual}")

        # 重用主菜单动作的图标 (假设它们已创建并带有图标)
        icon_show_tm = self.show_tm_action.icon()
        icon_show_ce = self.show_ce_action.icon()
        # --- 图标定义完毕 ---

        # --- 创建带图标的动作 ---
        # 移除动作
        remove_action = QAction(icon_remove, f"移除: {item_text}", self)  # <-- 添加图标
        # 正确连接 triggered 信号
        remove_action.triggered.connect(
            lambda checked=False, item_to_remove=item: self.remove_image_item(item_to_remove)
        )
        menu.addAction(remove_action)

        menu.addSeparator() # 添加分隔符

        # 显示模板匹配结果动作
        show_tm_action_ctx = QAction(icon_show_tm, "显示匹配结果", self)  # <-- 添加图标，注意变量名区分
        show_tm_action_ctx.triggered.connect(
            lambda checked=False, path=image_path: self.show_specific_image_result("tm", path)
        )
        # 可选：如果结果不存在则禁用 (可以添加检查)
        # if not self.processing_results.get(image_path, {}).get("tm_pixmap"):
        #     show_tm_action_ctx.setEnabled(False)
        menu.addAction(show_tm_action_ctx)

        # 显示坐标提取结果动作
        show_ce_action_ctx = QAction(icon_show_ce, "显示坐标提取结果", self)  # <-- 添加图标
        show_ce_action_ctx.triggered.connect(
            lambda checked=False, path=image_path: self.show_specific_image_result("ce", path)
        )
        # 可选：如果结果不存在则禁用
        # if not self.processing_results.get(image_path, {}).get("ce_pixmap"):
        #     show_ce_action_ctx.setEnabled(False)
        menu.addAction(show_ce_action_ctx)
        # --- 动作创建完毕 ---


        # 在请求点执行上下文菜单
        menu.exec(self.image_list_widget.mapToGlobal(point))

    def remove_image_item(self, item):
        """从列表中移除图像项及其相关数据。"""
        # ... (实现保持不变) ...
        if not item:
            return
        row = self.image_list_widget.row(item) # 获取项的行号
        item_text = item.text()
        image_path = self.image_files.pop(item_text, None) # 从字典中移除并获取路径
        if image_path:
            if image_path in self.processing_results:
                del self.processing_results[image_path] # 删除处理结果
            print(f"已移除图像: {item_text} ({image_path})")
            self.status_bar.showMessage(f"已移除图像: {item_text}")
            self.image_list_widget.takeItem(row) # 从列表控件中移除项
            if image_path == self.current_image_path: # 如果移除的是当前图像
                self.current_image_path = None
                self.image_scene.clear()
                self._reset_control_point_table() # 重置控制点表
                self.update_result_displays() # 更新结果显示
                if self.image_list_widget.count() > 0: # 如果列表不为空，则选中一个新项
                    self.image_list_widget.setCurrentRow(max(0, min(row, self.image_list_widget.count() - 1)))
                else: # 如果列表为空，则关闭所有结果窗口
                    self.close_all_result_windows()
        else:
            print(f"移除项 '{item_text}' 时出错，未找到路径。")

    def show_specific_image_result(self, window_type, image_path):
        """显示特定图像的存储结果（从右键菜单调用）。"""
        # *** 显示特定图像的存储结果 (从右键调用) ***
        # *** 重新创建时传递图标路径 *** <--- 修正说明
        if not image_path:
            return

        window = self.result_windows.get(window_type) # 获取窗口引用
        pixmap = None
        title = ""
        icon_rel_path = None  # <-- 初始化图标相对路径
        base_name = os.path.basename(image_path or "")

        # --- 检查并获取 Pixmap ---
        if window_type == 'plot':  # 绘图窗口由专门的 show_plot_window 处理
            self.show_plot_window()
            return
        elif image_path in self.processing_results:
            pixmap_key = f"{window_type}_pixmap"
            pixmap = self.processing_results[image_path].get(pixmap_key)
        else:  # image_path 不在结果字典中
            pixmap = None
        # --- Pixmap 获取结束 ---

        # --- 根据窗口类型确定标题和图标路径 ---
        if window_type == "tm":
            title = f"模板匹配 - {base_name}"
            icon_rel_path = "图标/icons8-缩略图-100.png"  # <-- TM 图标路径
        elif window_type == "ce":
            title = f"圆心提取 - {base_name}"
            icon_rel_path = "图标/icons8-缩略图-64.png"  # <-- CE 图标路径
        elif window_type == "ct": # 对应坐标转换可视化
            title = f"坐标转换结果 - {base_name}"
            icon_rel_path = "图标/icons8-组合图-48.png" # <-- 如果有 CT 图标路径
        else:
            self.status_bar.showMessage(f"未知窗口类型 '{window_type}'")
            return
        # --- 确定结束 ---

        # --- 创建或显示窗口 ---
        if not window:  # 如果窗口不存在或已关闭，创建/重新创建它
            if title:  # 确保有有效的标题
                print(f"通过右键为以下任务重新创建窗口: {window_type}")
                # --- 关键修正：创建时传递图标路径 ---
                window = ResultDisplayWindow(window_type, title, icon_rel_path=icon_rel_path)  # <-- 添加 icon_rel_path
                self.result_windows[window_type] = window
                window.windowClosed.connect(self.on_result_window_closed)
            else:  # 不应该发生，但作为保险
                self.status_bar.showMessage(f"无法为 '{window_type}' 创建窗口（无标题）")
                return

        # --- 更新并显示窗口 ---
        if pixmap:  # 如果有图像结果
            window.setWindowTitle(title)  # 确保标题正确
            window.set_image(pixmap, image_path) # 设置图像和源路径
            window.show()
            window.raise_()
        else:  # 如果没有图像结果 (可能是还没生成)
            # 如果窗口是新建的，显示提示；如果已存在，也显示提示
            self.status_bar.showMessage(f"图像 '{base_name}' 的 '{title}' 结果尚未生成。")
            # 也可以选择显示一个空的窗口并提示
            # window.setWindowTitle(title)
            # window.set_image(None, image_path) # 清空图像
            # window.show()
            # window.raise_()


    def update_comparison_plot_threaded(self):
        """在后台运行绘图任务，存储结果，不自动显示窗口。"""
        # *** 在后台运行绘图，存储结果，不自动显示窗口 ***
        self.status_bar.showMessage("正在后台更新多期对比图数据...")
        def plot_task(all_results_data, signals):
            """多期对比图的后台绘图任务函数。"""
            # ... [ 绘图计算逻辑与之前响应相同 ] ...
            signals.status_update.emit("多期对比图绘制线程开始...")
            import time
            time.sleep(0.1) # 模拟耗时
            try:
                periods_data = [] # 存储各期有效坐标数据
                # 过滤出有效的图像路径并排序
                sorted_paths = sorted(
                    [p for p in all_results_data if isinstance(p, str) and p != 'plot_pixmap'],
                    key=lambda p: os.path.basename(p)
                )
                for img_path in sorted_paths:
                    if 'real_coords' in all_results_data[img_path] and \
                       isinstance(all_results_data[img_path]['real_coords'], np.ndarray) and \
                       all_results_data[img_path]['real_coords'].size > 0:
                        periods_data.append(all_results_data[img_path]['real_coords'])

                num_time_steps = len(periods_data)
                signals.status_update.emit(f"找到 {num_time_steps} 期有效坐标数据...")
                plot_pixmap = QPixmap() # 初始化空的 QPixmap

                if num_time_steps >= 1: # 即使只有一期数据也允许绘图（显示初始点）
                    # ... (匹配、DataFrame、Delta 计算 - 相同) ...
                    tracked_data = [] # 存储追踪的点数据
                    initial_coords = {} # 存储第一期点的唯一 ID 和坐标
                    first_period_coords = periods_data[0]
                    # 为第一期数据分配唯一 ID
                    for i, coord in enumerate(first_period_coords):
                        unique_id = i # 使用索引作为唯一 ID
                        initial_coords[unique_id] = coord
                        tracked_data.append({'unique_id': unique_id, 'time_step': 1,
                                             'x_coord': coord[0], 'y_coord': coord[1]})

                    if num_time_steps >= 2: # 如果有多期数据，则进行点位匹配
                        signals.status_update.emit("正在匹配各期点位...")
                        reference_points = initial_coords.copy() # 参考点为上一期的匹配结果或初始点
                        MAX_MATCHING_DISTANCE = 0.5 # 最大匹配距离阈值 (单位与实际坐标一致)

                        for t in range(1, num_time_steps): # 从第二期开始
                            current_coords = periods_data[t] # 当前期坐标
                            if len(current_coords) == 0 or len(reference_points) == 0:
                                signals.status_update.emit(f"警告: 第 {t+1} 期无有效点或参考点丢失")
                                continue

                            ref_ids = list(reference_points.keys()) # 上一期参考点的 ID
                            last_coords = np.array([reference_points[uid] for uid in ref_ids]) # 上一期参考点的坐标

                            # 计算当前期点与上一期参考点之间的距离
                            distances = cdist(last_coords, current_coords)
                            potential_matches = [] # 存储可能的匹配 (距离, 参考点ID, 当前期点索引)
                            for ref_idx, uid in enumerate(ref_ids):
                                for cur_idx in range(current_coords.shape[0]):
                                    dist = distances[ref_idx, cur_idx]
                                    if dist <= MAX_MATCHING_DISTANCE:
                                        potential_matches.append((dist, uid, cur_idx))

                            potential_matches.sort() # 按距离排序
                            matched_cur_indices = set() # 已匹配的当前期点索引
                            matched_ref_ids = set() # 已匹配的参考点 ID
                            current_matches_dict = {} # 当前期匹配结果 {参考点ID: 当前期坐标}
                            new_reference_points = {} # 更新下一期的参考点

                            # 贪心匹配：选择距离最近的未匹配点对
                            for dist, uid, cur_idx in potential_matches:
                                if uid not in matched_ref_ids and cur_idx not in matched_cur_indices:
                                    coord = current_coords[cur_idx]
                                    current_matches_dict[uid] = coord
                                    new_reference_points[uid] = coord # 更新参考点
                                    matched_cur_indices.add(cur_idx)
                                    matched_ref_ids.add(uid)

                            signals.status_update.emit(f"第 {t+1} 期: 成功匹配 {len(matched_ref_ids)} / {len(ref_ids)} 个点")
                            # 将匹配结果添加到 tracked_data，未匹配的参考点用 NaN填充
                            for uid in ref_ids:
                                if uid in current_matches_dict:
                                    coord = current_matches_dict[uid]
                                    tracked_data.append({'unique_id': uid, 'time_step': t + 1,
                                                         'x_coord': coord[0], 'y_coord': coord[1]})
                                else: # 未匹配到
                                    tracked_data.append({'unique_id': uid, 'time_step': t + 1,
                                                         'x_coord': np.nan, 'y_coord': np.nan})
                            reference_points = new_reference_points # 更新参考点为当前期的匹配结果

                    if not tracked_data:
                        raise ValueError("点位追踪数据为空")

                    df = pd.DataFrame(tracked_data) # 创建 DataFrame
                    if 1 not in df['time_step'].values: # 确保第一期数据存在
                        raise ValueError("无法找到第一期数据")

                    # 获取第一期的 X, Y 坐标作为基准
                    initial_y = df.loc[df['time_step'] == 1].set_index('unique_id')['y_coord']
                    initial_x = df.loc[df['time_step'] == 1].set_index('unique_id')['x_coord']
                    if not initial_x.empty and not initial_y.empty:
                        initial_coords_df = pd.DataFrame({'x_initial': initial_x, 'y_initial': initial_y})
                        df = df.join(initial_coords_df, on='unique_id') # 合并初始坐标到 DataFrame
                    else: # 处理初始数据缺失的情况
                        df['x_initial'] = np.nan
                        df['y_initial'] = np.nan

                    df['delta_y'] = df['y_coord'] - df['y_initial'] # 计算 Y 坐标变化量 (高度变化)

                    # --- 生成组合图 ---
                    signals.status_update.emit("生成组合图...")
                    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 2]})
                    ax1 = axes[0] # 上部子图：高度变化时间序列

                    if num_time_steps >= 2 and 'delta_y' in df.columns and not df['delta_y'].isnull().all():
                        max_points_to_label = 15 # 最多标记的点数
                        points_labeled = 0
                        unique_ids = df['unique_id'].unique()
                        for uid in unique_ids:
                            subset = df[df['unique_id'] == uid].sort_values('time_step')
                            if not subset['delta_y'].isnull().all(): # 确保有有效的 delta_y 数据
                                label = f'点 {uid}' if points_labeled < max_points_to_label else None
                                ax1.plot(subset['time_step'], subset['delta_y'], marker='o', linestyle='-',
                                         markersize=4, label=label)
                                if label:
                                    points_labeled += 1
                        if points_labeled > 0 : # 只有在有标签时才显示图例
                            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                    elif num_time_steps < 2:
                        ax1.text(0.5, 0.5, '需要至少两期数据\n才能显示时间序列变化',
                                 ha='center', va='center', wrap=True)
                    else: # delta_y 全为 NaN
                        ax1.text(0.5, 0.5, '无法计算高度变化量', ha='center', va='center')

                    ax1.set_xlabel("时间期数")
                    ax1.set_ylabel("高度变化量 ($\\Delta Y$)") # 使用 LaTeX 格式化 Delta Y
                    ax1.set_title("监测点高度随时间的变化")
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # X轴刻度为整数

                    ax2 = axes[1] # 下部子图：最后一期高度变化空间分布
                    final_time_step = num_time_steps
                    # 准备用于散点图的数据，基于初始坐标和最后一期的 delta_y
                    initial_coords_df = pd.DataFrame({'x_initial': initial_x, 'y_initial': initial_y}).dropna() # 移除初始坐标缺失的点
                    if final_time_step in df['time_step'].values:
                        data_final_step = df[df['time_step'] == final_time_step].set_index('unique_id')
                        plot_data_final = initial_coords_df.join(data_final_step['delta_y'])
                        plot_data_final.dropna(subset=['delta_y'], inplace=True) # 仅基于 delta_y 移除 NaN
                    else: # 如果最后一期数据不存在
                        plot_data_final = pd.DataFrame(columns=['x_initial', 'y_initial', 'delta_y'])


                    if not plot_data_final.empty and not plot_data_final[['x_initial', 'y_initial']].isnull().all().all(): # 检查是否有有效的坐标数据
                        scatter = ax2.scatter(plot_data_final['x_initial'], plot_data_final['y_initial'],
                                              c=plot_data_final['delta_y'], cmap='coolwarm', s=50,
                                              edgecolors='k', linewidth=0.5)
                        cbar = fig.colorbar(scatter, ax=ax2)
                        cbar.set_label('高度变化量 ($\\Delta Y$)')
                        # 设置颜色条范围，使其对称且不为零
                        max_abs_delta = plot_data_final['delta_y'].abs().max()
                        if not pd.isna(max_abs_delta) and max_abs_delta > 1e-9: # 避免最大绝对值为 NaN 或非常接近零
                            clim_val = max(max_abs_delta, 0.001) # 最小范围为 0.001
                            scatter.set_clim(-clim_val, clim_val)
                    elif not initial_coords_df.empty: # 如果只有初始点，没有最终期数据或变化量
                        ax2.scatter(initial_coords_df['x_initial'], initial_coords_df['y_initial'],
                                    c='lightgray', s=50, label='无最终期数据或变化量')
                        ax2.legend()
                    else: # 无任何有效数据
                        ax2.text(0.5, 0.5, '无有效坐标数据可绘制', ha='center', va='center')

                    ax2.set_xlabel("X坐标")
                    ax2.set_ylabel("Y坐标")
                    ax2.set_title(f"第 {final_time_step} 期监测点高度变化的空间分布")
                    ax2.set_aspect('equal', adjustable='box') # 保持X,Y轴比例一致
                    ax2.grid(True, linestyle='--', alpha=0.6)

                    plt.tight_layout(pad=2.0) # 调整布局
                    # 将 matplotlib 图像保存到 BytesIO 对象
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=500) # 使用较高的 DPI
                    plt.close(fig) # 关闭图像以释放内存
                    buf.seek(0)
                    # 从 BytesIO 创建 QImage，然后创建 QPixmap
                    qimg = QImage.fromData(buf.getvalue(), 'PNG')
                    plot_pixmap = QPixmap.fromImage(qimg)
                else: # 数据不足
                    signals.status_update.emit("无法生成对比图（数据不足）")

                signals.result.emit({"plot_pixmap": plot_pixmap}) # 发出结果
                signals.status_update.emit("多期对比图数据已在后台更新")
            except Exception as e:
                import traceback
                print(f"绘图任务错误: {e}\n{traceback.format_exc()}")
                signals.status_update.emit(f"绘图错误: {e}")
                signals.result.emit({"error": "Plotting failed", "plot_pixmap": QPixmap()}) # 即使出错也发出空 pixmap

        results_copy = self.processing_results.copy() # 复制结果字典以避免线程冲突
        worker = Worker(plot_task, results_copy)
        thread = QThread()
        self.threadpool.append((thread, worker))
        worker.moveToThread(thread)
        worker.signals.result.connect(self.handle_plot_result) # 连接绘图结果处理槽函数
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.signals.status_update.connect(self.status_bar.showMessage)
        worker.signals.error.connect(lambda x: self.status_bar.showMessage(f"绘图线程错误: {x}"))
        thread.started.connect(worker.run)
        thread.start()
        worker.signals.finished.connect(self.hide_progress) # 任务完成后隐藏进度条


    def handle_plot_result(self, result_data):
        """处理多期对比图结果的槽函数。"""
        # ... (实现保持不变) ...
        if isinstance(result_data, dict):
            if "plot_pixmap" in result_data:
                self.processing_results['plot_pixmap'] = result_data["plot_pixmap"] # 存储绘图结果
                print("绘图 pixmap 已在后台更新/存储。")
                # 如果绘图窗口已打开且可见，则更新其内容
                plot_window = self.result_windows.get("plot")
                if plot_window and plot_window.isVisible() and not result_data["plot_pixmap"].isNull():
                    plot_window.set_image(result_data["plot_pixmap"])
                    print("已更新打开的绘图窗口。")
            elif "error" in result_data:
                self.status_bar.showMessage(f"后台绘图失败: {result_data['error']}")

    def update_result_displays(self):
        """更新主界面上显示处理结果的标签。"""
        self.tm_result_label.setText("匹配区域数量: N/A")
        self.ce_result_label.setText("提取中心数量: N/A")
        self.ct_result_label.setText("状态: 未转换")
        if self.current_image_path and self.current_image_path in self.processing_results:
            data = self.processing_results[self.current_image_path]
            if 'tm_count' in data:
                self.tm_result_label.setText(f"匹配区域数量: {data['tm_count']}")
            if 'ce_count' in data:
                self.ce_result_label.setText(f"提取中心数量: {data['ce_count']}")
            if 'real_coords' in data:
                num_transformed = len(data['real_coords']) if isinstance(data['real_coords'], np.ndarray) else 0
                self.ct_result_label.setText(f"状态: {num_transformed} 点已转换")

    def hide_progress(self):
        """如果所有线程都已完成，则隐藏进度条。"""
        all_finished = True
        active_threads = [] # 用于存储仍在运行的线程
        for thread, worker in self.threadpool:
            if thread is not None and worker is not None and thread.isRunning():
                all_finished = False
                active_threads.append((thread, worker)) # 保留活动线程
        if all_finished:
            self.progress_bar.hide()
        self.threadpool = active_threads # 更新线程池，移除已完成的线程

    def closeEvent(self, event):
        """处理主窗口关闭事件。"""
        self.close_all_result_windows() # 关闭所有子窗口
        print("正在退出应用程序...")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # --- 2. 应用 QDarkStyleSheet 主题 ---
    if qdarkstyle: # 检查库是否成功导入
        try:
            # 获取并应用 QDarkStyleSheet (适用于 qdarkstyle v3+)
            # 对于旧版本或特定绑定，可能需要 qdarkstyle.load_stylesheet_pyside6()
            stylesheet = qdarkstyle.load_stylesheet()
            app.setStyleSheet(stylesheet)
        except Exception as e:
            print(f"错误: 应用 qdarkstyle 主题失败 - {e}")
            # 可以选择保留默认样式或尝试其他备用方案
    # else: # 如果 qdarkstyle 未导入，则不执行任何操作，使用默认样式
    #     pass
    # --- 主题应用结束 ---




    main_win = MainWindow() # 创建主窗口实例

    # ... (设置应用程序图标的代码) ...
    icon_rel_path = "图标/icons8-图片编辑器-48.png"
    icon_path = resource_path(icon_rel_path)  # <--- 使用辅助函数获取绝对路径
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        main_win.setWindowIcon(app_icon) # 设置主窗口图标
    else:
        print(f"警告: 应用程序图标文件未找到: {icon_path} (原始相对路径: {icon_rel_path})")

    main_win.show() # 显示主窗口
    sys.exit(app.exec()) # 进入应用程序事件循环
