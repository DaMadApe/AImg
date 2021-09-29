
# Módulos de la GUI
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
# Módulos adicionales
import cv2
import numpy as np
import imglib as lib


class MainContainer(BoxLayout):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_file_popup = Load_file_popup(load=self.load)

    def slideCompare(self, instance, val):
        self.ids.sliderVal.text = "% d"% val

    def cargarImagen(self):
        img = cv2.imread('datos/panda.jpg', cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        img_eq = np.ones(img.shape, dtype=np.uint8)
        img_canvas = np.zeros(img.shape, dtype=np.uint8)
        h, w, _ = img.shape
        imgView = Texture.create(size=(w, h))
        imgView.blit_buffer(img_canvas.flatten(), colorfmt='rgb', bufferfmt='ubyte')

    def callback_load(self, *args):
        self.load_file_popup.open()

    def load(self, selection):
        self.file_path = str(selection[0])
        self.ids.pathLabel.text = self.file_path
        self.dismiss()


class BetterImgApp(App):
    def build(self):
        container = MainContainer()
        return container


class Load_file_popup(Popup):
    load = ObjectProperty()


if __name__ == '__main__':
    Window.size = (480, 800)
    root = BetterImgApp()
    root.run()