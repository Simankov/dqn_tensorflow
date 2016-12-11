#!/usr/bin/env python

import pygame
from sys import exit
import pygame.constants

class Pong:
    def __init__(self):
        self.paused = False
        self.bar1_score = 0
        self.bar2_score = 0
    def start(self):
        pygame.init()
        ai_speed = 1000
        screen=pygame.display.set_mode((640,480),0,32)
        pygame.display.set_caption("Pong Pong!")
        # Creating 2 bars, a ball and background.
        back = pygame.Surface((640, 480))
        background = back.convert()
        background.fill((0, 0, 0))
        bar = pygame.Surface((10, 50))
        bar1 = bar.convert()
        bar1.fill((0, 0, 255))
        bar2 = bar.convert()
        bar2.fill((255, 0, 0))
        circ_sur = pygame.Surface((15, 15))
        circ = pygame.draw.circle(circ_sur, (0, 255, 0), (15 / 2, 15 / 2), 15 / 2)
        circle = circ_sur.convert()
        circle.set_colorkey((0, 0, 0))
        # some definitions
        bar1_x, bar2_x = 10., 620.
        bar1_y, bar2_y = 215., 215.
        circle_x, circle_y = 307.5, 232.5
        bar1_move, bar2_move = 0., 0.
        speed_x, speed_y, speed_circ = 250., 250., 250.
        # clock and font objects
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("calibri", 40)

        while True:

            for event in pygame.event.get():
                if isinstance(event, int):
                    continue
                if event.type == QUIT:
                    exit()
                if event.type == KEYDOWN:
                    if event.key == K_UP:
                        bar1_move = -ai_speed
                    elif event.key == K_DOWN:
                        bar1_move = ai_speed
                elif event.type == KEYUP:
                    if event.key == K_UP:
                        bar1_move = 0.
                    elif event.key == K_DOWN:
                        bar1_move = 0.

            score1 = font.render(str(self.bar1_score), True, (255, 255, 255))
            score2 = font.render(str(self.bar2_score), True, (255, 255, 255))

            screen.blit(background, (0, 0))
            frame = pygame.draw.rect(screen, (255, 255, 255), Rect((5, 5), (630, 470)), 2)
            middle_line = pygame.draw.aaline(screen, (255, 255, 255), (330, 5), (330, 475))
            screen.blit(bar1, (bar1_x, bar1_y))
            screen.blit(bar2, (bar2_x, bar2_y))
            screen.blit(circle, (circle_x, circle_y))
            screen.blit(score1, (250., 210.))
            screen.blit(score2, (380., 210.))

            bar1_y += bar1_move
            pygame.init()

            # movement of circle
            time_passed = clock.tick(30)
            time_sec = time_passed / 1000.0
            dx = 0
            dy = 0
            if not self.paused:
                dx = speed_x * time_sec
                dy = speed_y * time_sec
                speed_circ = 250.
            else:
                speed_circ = 0
                dx = 0
                dy = 0

            circle_x += dx
            circle_y += dy
            ai_speed = speed_circ * time_sec
            # AI of the computer.
            if circle_x >= 305.:
                if not bar2_y == circle_y + 7.5:
                    if bar2_y < circle_y + 7.5:
                        bar2_y += ai_speed
                    if bar2_y > circle_y - 42.5:
                        bar2_y -= ai_speed
                else:
                    bar2_y == circle_y + 7.5

            if bar1_y >= 420.:
                bar1_y = 420.
            elif bar1_y <= 10.:
                bar1_y = 10.
            if bar2_y >= 420.:
                bar2_y = 420.
            elif bar2_y <= 10.:
                bar2_y = 10.
            # since i don't know anything about collision, ball hitting bars goes like this.
            if circle_x <= bar1_x + 10.:
                if circle_y >= bar1_y - 7.5 and circle_y <= bar1_y + 42.5:
                    circle_x = 20.
                    speed_x = -speed_x
            if circle_x >= bar2_x - 15.:
                if circle_y >= bar2_y - 7.5 and circle_y <= bar2_y + 42.5:
                    circle_x = 605.
                    speed_x = -speed_x
            if circle_x < 5.:
                self.bar2_score += 1
                circle_x, circle_y = 320., 232.5
                bar1_y, bar_2_y = 215., 215.
            elif circle_x > 620.:
                self.bar1_score += 1
                circle_x, circle_y = 307.5, 232.5
                bar1_y, bar2_y = 215., 215.
            if circle_y <= 10.:
                speed_y = -speed_y
                circle_y = 10.
            elif circle_y >= 457.5:
                speed_y = -speed_y
                circle_y = 457.5

            pygame.display.update()

    def switch_pause(self):
        self.paused = not self.paused

    def get_scores(self):
        return (self.bar1_score,self.bar2_score)