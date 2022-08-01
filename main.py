"""
Estimate Relaxation from Band Powers
This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.
The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al
Adapted from https://github.com/NeuroTechX/bci-workshop
"""
import pygame
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import utils


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear

INDEX_CHANNEL = [0]
streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')

# Set active EEG stream to inlet and apply time correction
inlet = StreamInlet(streams[0], max_chunklen=12)
eeg_time_correction = inlet.time_correction()

# Get the stream info and description
info = inlet.info()
description = info.desc()

# Get the sampling frequency
# This is an important value that represents how many EEG data points are
# collected in a second. This influences our frequency band calculation.
# for the Muse 2016, this should always be 256
fs = int(info.nominal_srate())


class Flag:
    # valuable identity, it is used to differentiate two players
    def __init__(self, screen):
        self.screen = screen
        self.top = 670
        self.img = pygame.image.load("flag.jpg").convert()
        self.img = pygame.transform.scale(self.img, (130, 130))
        self.speed = 8

    def move(self):
        """ 2. INITIALIZE BUFFERS """

        # Initialize raw EEG data buffer
        eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
        filter_state = None  # for use with the notch filter

        # Compute the number of epochs in "buffer_length"
        n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                  SHIFT_LENGTH + 1))

        # Initialize the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        band_buffer = np.zeros((n_win_test, 4))

        """ 3.1 ACQUIRE DATA """
        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))

        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

        # Update EEG buffer with the new data
        eeg_buffer, filter_state = utils.update_buffer(
            eeg_buffer, ch_data, notch=True,
            filter_state=filter_state)

        """ 3.2 COMPUTE BAND POWERS """
        # Get newest samples from the buffer
        data_epoch = utils.get_last_data(eeg_buffer,
                                         EPOCH_LENGTH * fs)

        # Compute band powers
        band_powers = utils.compute_band_powers(data_epoch, fs)
        band_buffer, _ = utils.update_buffer(band_buffer,
                                             np.asarray([band_powers]))
        # Compute the average band powers for all epochs in buffer
        # This helps to smooth out noise
        smooth_band_powers = np.mean(band_buffer, axis=0)

        """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
        # Beta Protocol:
        # Beta waves have been used as a measure of mental activity and concentration
        # This beta over theta ratio is commonly used as neurofeedback for ADHD
        beta_metric = smooth_band_powers[Band.Beta] / \
                      smooth_band_powers[Band.Theta]
        print("Beta_metric:", beta_metric)
        ###########################################
        # the bigger beta_metric means the more concentrates
        # the flag raise when player is concentrates,and goes down if not
        if beta_metric > 2 and self.top > 10:
            self.top -= self.speed
        elif self.top < 660:
            self.top += self.speed

    # A method to display the flag
    def draw(self):
        self.screen.blit(self.img, (370, self.top))

    # A method to get how high the flag currently is
    def getTop(self):
        return self.top


class Game:
    def __init__(self, surface):
        self.screen = surface
        self.background = pygame.image.load("playground.png").convert()
        self.background = pygame.transform.scale(self.background, (1000, 800))
        self.game_clock = pygame.time.Clock()
        self.FPS = 60
        self.flag = Flag(self.screen)
        self.gameRunning = True
        self.close_clicked = False

        pygame.mixer.init()
        pygame.mixer.music.load('SweetBGM.mp3')
        pygame.mixer.music.play()

    def handle_events(self):
        events = pygame.event.get()
        # end the game is player click "close" button
        for event in events:
            # once click close, self.close_clicked =true
            if event.type == pygame.QUIT:
                self.close_clicked = True

    def play(self):
        while (not self.close_clicked) and self.flag.getTop() > 50:
            self.handle_events()
            self.screen.fill('black')
            self.screen.blit(self.background, (0, 0))
            pygame.draw.circle(self.screen, pygame.Color('black'), (360, 40), 10)
            pygame.draw.rect(self.screen, pygame.Color('black'), pygame.Rect(350, 50, 20, 790))
            self.flag.move()
            self.flag.draw()
            pygame.display.flip()
            self.game_clock.tick(self.FPS)
        if self.flag.getTop() <= 50:
            self.afterWin()

    def afterWin(self):
        # if player reach the top, the game stop and show "You win!!"
        while not self.close_clicked:
            self.handle_events()
            self.flag.draw()
            self.screen.blit(
                pygame.font.SysFont('', 60, bold=True).render(str("You Win!!"), True, pygame.Color('white')), (50, 50))
            pygame.display.flip()
            self.game_clock.tick(self.FPS)
            if self.close_clicked:
                pygame.quit()


def main():
    pygame.init()
    size = (1000, 800)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("FlagGame")
    game = Game(screen)
    game.play()
    pygame.quit()


main()
