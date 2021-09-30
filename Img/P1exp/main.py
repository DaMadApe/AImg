
# Módulos de la GUI
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.layout import Layout
from kivy.graphics import Rectangle
from kivy.graphics.instructions import Callback
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
# Módulos adicionales
import os
import cv2
import numpy as np
import imglib as lib


class MainContainer(BoxLayout):
    
    def __init__(self, **kwargs):
        super(MainContainer, self).__init__(**kwargs)

    def slideCompare(self, instance, val):
        self.ids.imgWidget.slideView(val)

    def load_popup(self):
        self.load_file_popup = Load_file_popup(load=self.load)
        self.load_file_popup.open()

    def save_popup(self):
        self.save_file_popup = Save_file_popup(save=self.save)
        self.save_file_popup.open()

    def load(self, selection):
        self.file_path = str(selection[0])
        self.ids.pathLabel.text = self.file_path
        self.load_file_popup.dismiss()
        self.ids.btn1.disabled = False
        self.ids.btn2.disabled = False
        self.ids.btn3.disabled = False
        self.ids.imgWidget.cargarImagen(self.file_path)

    def save(self, path, filename):
        save_path = os.path.join(path, filename)
        self.save_file_popup.dismiss()
        self.ids.imgWidget.guardarImagen(save_path)

    def selecEcualizador(self, n):
        func = [lib.ecu_hist, lib.clahe, lib.m3][n]
        self.ids.viewSlider.disabled = False
        self.ids.btnSave.disabled = False
        self.ids.imgWidget.ecualizar(func)


class TexView(Layout):

    def __init__(self, **kwargs):
        super(TexView, self).__init__(**kwargs)
        self.initCanvas()

    def initCanvas(self):
        black_canvas = np.zeros((10,10,3), dtype=np.uint8)
        self.tex = Texture.create(size=(10, 10))
        self.tex.blit_buffer(black_canvas.flatten(), colorfmt='rgb', bufferfmt='ubyte')
        with self.canvas:
            self.rect = Rectangle(texture=self.tex,
                                  size=(self.width, self.height*2),
                                  pos=(self.center_x -self.width/2,
                                       self.center_y -self.height/2))

    def update(self):
        self.cb.ask_update()

    def cargarImagen(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(img.shape)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = cv2.flip(img, 0)
        self.img_canvas = np.zeros(self.img.shape, dtype=np.uint8)
        h, w, _ = self.img.shape
        self.tex = Texture.create(size=(w, h))
        self.tex.blit_buffer(self.img.flatten(), colorfmt='bgr', bufferfmt='ubyte')
        with self.canvas:
            self.rect = Rectangle(texture=self.tex,
                                  size=(self.width, self.height*2),
                                  pos=(self.center_x -self.width/2,
                                       self.center_y -self.height/2))
            self.cb = Callback()

    def guardarImagen(self, path):
        cv2.imwrite(path, cv2.flip(self.img_eq, 0))

    def ecualizar(self, ecu_fun):
        self.img_eq = ecu_fun(self.img)
        self.slideView(0)

    def slideView(self, n):
        idx = int(self.img.shape[1] * n/100)
        self.img_canvas[:,idx:,:] = self.img[:,idx:,:]
        self.img_canvas[:,:idx,:] = self.img_eq[:,:idx,:]
        self.tex.blit_buffer(self.img_canvas.flatten(), colorfmt='bgr', bufferfmt='ubyte')
        self.update()
    
    def ponerImg(self):
        self.cargarImagen('datos/panda.jpg')


class Load_file_popup(Popup):
    load = ObjectProperty()

class Save_file_popup(Popup):
    save = ObjectProperty()



class BetterImgApp(App):
    def build(self):
        container = MainContainer()
        #container.cargarImagen()
        return container



if __name__ == '__main__':
    Window.size = (480, 800)
    root = BetterImgApp()
    root.run()