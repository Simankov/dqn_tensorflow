import pygame
paused = False
def switch_pause():
    global paused
    paused = not paused
    if paused:
        pygame.time.delay(5000)