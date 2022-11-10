#!/usr/bin/env python
""" pg.examples.stars

    We are all in the gutter,
    but some of us are looking at the stars.
                                            -- Oscar Wilde

A simple starfield example. Note you can move the 'center' of
the starfield by leftclicking in the window. This example show
the basics of creating a window, simple pixel plotting, and input
event management.
"""
import random
import math, sys, os
import pygame as pg

# constants
WINSIZE = [640, 480]
WINCENTER = [320, 240]

with open("data/zip/zip.train") as f:
    for i in range(101):
        l = f.readline()
l = [float(x) for x in l.strip("\n").split(" ") if x]
print(l[0])
l = l[1:]
#$os._exit(0)

def main():
    "This is the starfield code"
    # create our starfield
    random.seed()
    clock = pg.time.Clock()
    # initialize and prepare screen
    pg.init()
    screen = pg.display.set_mode(WINSIZE)
    pg.display.set_caption("pygame Stars Example")
    white = 255, 240, 200
    black = 20, 20, 40
    screen.fill(black)
    surf = pg.display.get_surface()
    # main game loop
    done = 0
    while not done:
        pg.display.update()
        for x in range(16):
            for y in range(16):
                try:
                    v = int((1 - l[y * 16 + x]) * 127)
                    pg.draw.rect(surf, pg.Color(v, v, v), pg.Rect(x * 40, y * 30, 40, 30))
                except ValueError:
                    import pdb
                    pdb.set_trace()
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYUP and e.key == pg.K_ESCAPE):
                done = 1
                break
            elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                WINCENTER[:] = list(e.pos)
        clock.tick(50)
    pg.display.quit()
    os._exit(0)
    sys.exit(0)
    pg.quit()


# if python says run, then we should run
if __name__ == "__main__":
    main()
