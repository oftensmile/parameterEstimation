#!/usr/bin/python
# coding: utf-8

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL import GL
def draw():
    # OpenGLバッファのクリア
    glClearColor(0.0, 0.5, 0.5, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    dw,dh,dl,dd=0.01,0.26,0.3,0.02
    bound=-0.8
    #GL.glRectf(-0.1,-0.1,0,0)#x1,y1,x2,y2
    #GL.glRectf(0,0,1,1)#x1,y1,x2,y2
    d=8
    for i1 in range(d+1):
        for i2 in range(d+1):
            if(i1<d):GL.glRectf(bound+dl*i1,bound+dd+i2*dl,bound+dl*i1+dw,bound-dd+dh+i2*dl)#x1,y1,x2,y2
            if(i2<d):GL.glRectf(bound+2*dd+dl*i1,bound+i2*dl,bound+-dd+dl*i1+dh,bound+i2*dl+dw)#x1,y1,x2,y2
    #glBegin(GL_TRIANGLES)
    #glVertex(-1,-1)
    #glVertex(1,-1)
    #glVertex(0,1)
    #glEnd()
    # OpenGL描画実行
    glFlush()
    # glutダブルバッファ交換
    glutSwapBuffers()


def setup():
    glutInit(sys.argv)
    # RGBAモード、ダブルバッファリング有効、Zバッファ有効で初期化
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow("glut sample")
    #this function is called when window size will change 
    glutReshapeFunc(resize)
    # 描画時に呼ばれる関数を登録
    glutDisplayFunc(draw)


def resize(w,h):
    print("resize",w,h)
    glViewport(50,150,w/2,h/2)


if __name__=="__main__":
    setup()
    glutMainLoop()
