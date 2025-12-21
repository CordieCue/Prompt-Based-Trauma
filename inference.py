import os
import csv
import argparse
import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from scipy.io.wavfile import write as write_wav

trauma_inputs= [
    "My head hurts. My leg is cut.",
    "Chest has big bruise. My arm is gone.",
    "Face is bleeding. Tummy hurts bad.",
    "My leg is trapped. My arm has a cut.",
    "Head is okay. My foot is missing.",
    "Arm is broke. My leg also broke.",
    "Tummy has glass in it. Head is bleeding.",
    "My chest is pinned. My head is okay.",
    "Leg is gone. My arm is also gone.",
    "My hand is stuck. My leg hurts.",
    "Face is cut. I cant feel my arm.",
    "My leg is numb. My chest hurts.",
    "Head is fine. My arm is trapped.",
    "Belly hurts. My foot is gone.",
    "Leg has a deep cut. Arm is bleeding.",
    "My head is bleeding. My leg is gone.",
    "Chest is okay. My arm is missing.",
    "My leg is stuck. My head hurts bad.",
    "Arm is numb. My leg is also numb.",
    "Tummy is cut. My hand is gone.",
    "My head is fine. My leg is stuck.",
    "Belly hurts bad. My hand is gone.",
    "Face is cut. My leg feels numb.",
    "My chest is crushed. My arm is broke.",
    "No hurt on head. My foot is trapped.",
    "Tummy has a big cut. Leg is bleeding.",
    "My head is bleeding. My chest hurts.",
    "Arm is gone. Leg is also gone.",
    "My hand is bleeding. Leg is broke.",
    "I cant feel my leg. My head hurts.",
    "My head hurts. My leg is gone.",
    "Belly is trapped. My arm is bleeding.",
    "Face is okay. My arm cant feel.",
    "My chest is bleeding. My leg is broke.",
    "No hurt on head. My foot is stuck.",
    "Tummy is fine. My hand is missing.",
    "My head is bleeding. My arm is gone.",
    "Leg is gone. My chest is trapped.",
    "My arm has a big cut. Leg is numb.",
    "I cant feel my head. My belly hurts.",
    "Head is bleeding. My leg is stuck.",
    "My chest hurts. My hand is missing.",
    "Face is fine. My whole leg feels numb.",
    "My belly is cut. My arm is bleeding.",
    "No hurt on head. My foot is gone.",
    "Tummy is okay. My arm is trapped.",
    "My head hurts. My hand is crushed.",
    "Arm is gone. My chest is bleeding.",
    "My arm is broke. My leg is gone.",
    "I cant feel my arm. My belly is cut.",
    
    "Big cut on my head, my chest is pinned under something heavy, my legs feel okay.",
    "My face is fine, but my belly is cut open. One arm is gone from the shoulder.",
    "My skull feels cracked, there's a big bruise on my ribs, and my leg is trapped.",
    "Stomach is bleeding a lot, and my arm... it's just gone. My other arm is fine.",
    "Head is okay, but my whole chest feels crushed. I cant feel my leg, it's totally numb.",
    "There's blood coming from my ear, and my arm is bent the wrong way. My leg is also broken.",
    "My neck is stuck, can't move my head. There is a deep cut in my side. My legs are fine.",
    "My head and belly hurt bad. My leg is gone below the knee. My arm is pinned.",
    "Face is all bloody, my ribs are snapped. My arm is fine but my leg is missing.",
    "No hurt on head. My chest is bleeding. I cant feel my arm at all.",
    "My head is okay, but my back is trapped under a wall. My arm has a bad cut.",
    "Face is bleedin, stomach feels okay. One leg gone, the other one is broke.",
    "Big cut on my head. Chest is fine. I cant feel my arm, it is dead.",
    "A hole in my stomach, my leg is twisted all wrong. My head is okay.",
    "My whole chest is crushed, and my arm is gone. My legs feel fine.",
    "My jaw is broke and my belly is cut open. My legs are stuck under metal.",
    "There is glass in my side. My arm is gone. My head feels okay.",
    "My head and ribs hurt a lot. My hand is gone. My leg is pinned down.",
    "Face is cut bad, my stomach is trapped. My arm is fine but my leg is gone.",
    "No hurt on head. My chest has a deep wound. My arm is gone, and so is my leg.",
    "Head is bleedin, my torso is stuck. Arms are okay.",
    "Face is fine. Chest has a big cut. My leg is gone below the knee.",
    "Big gash on my head. Ribs are okay. My whole arm is all numb.",
    "Stomach is bleedin, my leg is twisted. My head has a big bruise.",
    "My chest is crushed, and one arm is gone. My legs feel weird and tingly.",
    "My jaw is broken and my stomach is cut. My legs are pinned under a seat.",
    "There is glass in my side. My arm is gone. My head feels fine.",
    "My head and belly hurt. My hand is gone. My leg is stuck.",
    "Face is all bloody, my ribs are broken. My arm is fine but my leg is gone.",
    "No hurt on my head. My chest is bleeding. I cant feel my arm, and my leg is gone.",
    "Head okay, but my chest is trapped. My leg is bleedin bad.",
    "Face is cut up, stomach is fine. One arm is gone, the other arm is broke.",
    "Big gash on my head. Ribs are fine. I cant feel my leg at all.",
    "A pipe is in my stomach, my arm is twisted. My head is fine.",
    "My whole chest hurts, and my arm is gone. My other arm is pinned down.",
    "My jaw is broke and my belly is cut. My legs... they are just gone.",
    "There is glass in my back. My leg is gone. My head feels fine.",
    "My head and stomach hurt bad. My arm is gone. My leg is also gone.",
    "Face is all bloody, my chest is trapped. My arm is okay but my leg is broken.",
    "No hurt on head. My belly is bleeding. My arm is numb, and my leg is trapped.",
    "My head is bleeding, and my torso is stuck. My arms are okay.",
    "My face is fine. My chest has a big cut, and my leg is gone below the knee.",
    "Big gash on my head. My ribs feel okay. My whole arm is just numb.",
    "My stomach is bleeding, and my leg is twisted. My head has a big bruise.",
    "My chest is crushed, and one of my arms is gone. My legs feel weird.",
    "My jaw is broken, and my stomach is cut. My legs are pinned under a seat.",
    "There is glass in my side. My arm is gone. My head feels fine.",
    "My head and my belly both hurt a lot. My hand is gone. My leg is stuck.",
    "My face is all bloody, and my ribs are broken. My arm is fine, but my leg is gone.",
    "No hurt on my head. My chest is bleeding. I cant feel my arm, and my leg is gone.",

    # "My face is gone, I can feel bone. My chest is pinned. One arm is gone, other is stuck. I cant feel my legs.",
    # "Head okay I think. My whole body is crushed. My left hand is gone. My right arm is numb. I see my legs are also gone.",
    # "Blood in my eyes, cant see. Big hole in my stomach. Both my arms are numb. My legs are trapped under the wreck.",
    # "My skull feels like it is open. My guts are hanging out. My arms are gone. My legs are also gone.",
    # "No head pain but my jaw is sideways. My torso is trapped, cant breathe right. One arm is gone. My legs feel like they on fire.",
    # "My head is under water, I cant hear. My back is snapped. My arms are pinned. My legs... they are not there anymore.",
    # "I can see my brain, my head is open. My chest has a pole through it. I cant feel my arms. My legs are twisted and broken.",
    # "Everything is black, cant see. My chest and belly are crushed. One arm is gone, other is numb. One leg gone, other leg numb.",
    # "My neck is broken, I cant move my head. My chest is fine. Both arms are gone. Both legs are gone.",
    # "My head is fine. But my body is stuck. I see my hands are missing, but my arms are there. My legs are pinned and I cant feel them.",
    # "My head is smashed, cant see. There is a pipe in my chest. Both arms are gone. My legs feel like nothing, just numb.",
    # "Head okay I think, but body is crushed, cant see it. My left arm is gone but my right arm is bleeding. Both my legs are missing too.",
    # "My face is all blood. My chest is pinned down. My left arm is gone, my right arm is numb. Both my legs are trapped.",
    # "My skull feels empty, it's all wet. My guts are outside my body. My arms are fine. My legs are gone.",
    # "No head pain. My torso is on fire. One arm is gone. The other is pinned. My legs are twisted and I cant feel them.",
    # "My head is under something, I cant breathe. My back is broken. My arms are gone. My legs are stuck.",
    # "I can see my own skull bone, my head is open. My chest has a big hole. I cant feel my arms. My legs are gone.",
    # "Everything is dark. My chest and belly are gone, I see inside. One arm is gone, other is numb. One leg gone, other leg numb.",
    # "My neck is broke. My chest is okay. Both arms are gone. My legs are also gone, but I cant see.",
    # "My head is fine. But my whole body is stuck. My hands are gone, but my arms are still here. My legs are pinned and I cant feel them.",
    # "My head is smashed, I cant see. There is a pipe in my chest. Both my arms are gone. My legs feel like nothing, just numb.",
    # "Head okay I think, but my body is crushed, I cant see it. My left arm is gone but my right arm is bleeding. Both my legs are missing too.",
    # "My face is all blood. My chest is pinned down. My left arm is gone, my right arm is numb. Both my legs are trapped.",
    # "My skull feels empty, it's all wet. My guts are outside my body. My arms are fine. My legs are gone.",
    # "No head pain. My torso is on fire. One arm is gone. The other is pinned. My legs are twisted and I cant feel them.",
    # "My head is under something, I cant breathe. My back is broken. My arms are gone. My legs are stuck.",
    # "I can see my own skull bone, my head is open. My chest has a big hole. I cant feel my arms. My legs are gone.",
    # "Everything is dark. My chest and belly are gone, I see inside. One arm is gone, other is numb. One leg gone, other leg numb.",
    # "My neck is broke. My chest is okay. Both arms are gone. My legs are also gone, but I cant see.",
    # "My head is fine. But my whole body is stuck. My hands are gone, but my arms are still here. My legs are pinned and I cant feel them.",
    # "My head is smashed, I cant see. There is a pipe in my chest. Both my arms are gone. My legs feel like nothing, just numb.",
    # "Head okay I think, but my body is crushed, I cant see it. My left arm is gone but my right arm is bleeding. Both my legs are missing too.",
    # "My face is all blood. My chest is pinned down. My left arm is gone, my right arm is numb. Both my legs are trapped.",
    # "My skull feels empty, it's all wet. My guts are outside my body. My arms are fine. My legs are gone.",
    # "No head pain. My torso is on fire. One arm is gone. The other is pinned. My legs are twisted and I cant feel them.",
    # "My head is under something, I cant breathe. My back is broken. My arms are gone. My legs are stuck.",
    # "I can see my own skull bone, my head is open. My chest has a big hole. I cant feel my arms. My legs are gone.",
    # "Everything is dark. My chest and belly are gone, I see inside. One arm is gone, other is numb. One leg gone, other leg numb.",
    # "My neck is broke. My chest is okay. Both arms are gone. My legs are also gone, but I cant see.",
    # "My head is fine. But my whole body is stuck. My hands are gone, but my arms are still here. My legs are pinned and I cant feel them.",
    # "My head is smashed, I cant see. There is a pipe in my chest. Both my arms are gone. My legs feel like nothing, just numb.",
    # "Head okay I think, but my body is crushed, I cant see it. My left arm is gone but my right arm is bleeding. Both my legs are missing too.",
    # "My face is all blood. My chest is pinned down. My left arm is gone, my right arm is numb. Both my legs are trapped.",
    # "My skull feels empty, it's all wet. My guts are outside my body. My arms are fine. My legs are gone.",
    # "No head pain. My torso is on fire. One arm is gone. The other is pinned. My legs are twisted and I cant feel them.",
    # "My head is under something, I cant breathe. My back is broken. My arms are gone. My legs are stuck.",
    # "I can see my own skull bone, my head is open. My chest has a big hole. I cant feel my arms. My legs are gone.",
    # "Everything is dark. My chest and belly are gone, I see inside. One arm is gone, other is numb. One leg gone, other leg numb.",
    # "My neck is broke. My chest is okay. Both arms are gone. My legs are also gone, but I cant see.",
    # "My head is fine. But my whole body is stuck. My hands are gone, but my arms are still here. My legs are pinned and I cant feel them."
]

trauma_labels = [
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",

    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NOT TESTABLE, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",

    # "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: NOT TESTABLE, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    # "Head: WOUND, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: WOUND, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: AMPUTATION",
    # "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE"
]
# --------------------------------------------------------
# Audio Functions
# --------------------------------------------------------
def record_audio(duration=10, sample_rate=16000):
    """Record audio from microphone."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def save_audio(audio_data, sample_rate, filename):
    """Save recorded audio to a WAV file."""
    audio_int16 = np.int16(audio_data * 32767)
    write_wav(filename, sample_rate, audio_int16)
    print(f"Audio saved to {filename}")

# --------------------------------------------------------
# Model Loading Functions
# --------------------------------------------------------
def load_whisper(whisper_model_dir, device="cpu"):
    """Load Whisper ASR pipeline."""
    return pipeline(
        "automatic-speech-recognition",
        model=whisper_model_dir,
        device=0 if device == "cuda" else -1
    )

def load_t5_custom(model_dir, device="cpu"):
    """Load custom fine-tuned T5 model and tokenizer."""
    tokenizer = T5Tokenizer.from_pretrained(
        "/media/vansh/Data/projects/pbt/allpbtfile/checkpoint-2550"
    )
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    return tokenizer, model

# --------------------------------------------------------
# Inference + Logging
# --------------------------------------------------------
def run_all_inferences(whisper_pipe, tokenizer, model, args, device):
    """Run inference for all trauma sentences and log results."""
    log_file = args.log_file
    error_log_file = log_file.replace(".csv", "_errors.csv")

    # Create CSV headers if they don't exist
    if not os.path.exists(log_file):
        with open(log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Index", "Prompted Sentence", "Transcription",
                "Predicted Label", "Ground Truth", "Exact Match"
            ])

    if not os.path.exists(error_log_file):
        with open(error_log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Sentence", "Ground Truth"])

    # Iterate through trauma dataset
    for idx, (true_sentence, true_label) in enumerate(zip(trauma_inputs, trauma_labels)):
        print(f"\nRunning inference on example {idx + 1}/{len(trauma_inputs)}:")

        input_text = (
            f"Describe any injuries you have on your head, torso, arms, or legs. "
            f"Be specific: {true_sentence}"
        )
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        ).to(device)

        # Model inference
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check match
        is_correct = predicted_label.strip() == true_label.strip()

        # Print debug info
        print(f"Sentence: {true_sentence}")
        print(f"Predicted: {predicted_label}")
        print(f"Ground Truth: {true_label}")
        print(f"Exact Match: {'Correct' if is_correct else 'Incorrect'}")

        # Write to main log
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                idx, true_sentence, true_sentence,
                predicted_label.strip(), true_label.strip(), int(is_correct)
            ])

        # Write to error log if mismatch
        if not is_correct:
            with open(error_log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([true_sentence, true_label])

# --------------------------------------------------------
# Main
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch trauma inference test")
    parser.add_argument("--whisper_model", type=str,
                        default="/media/vansh/Data/projects/pbt/allpbtfile/temp_dir_1",
                        help="Path to Whisper model directory")
    parser.add_argument("--t5_model_path", type=str,
                        default="/media/vansh/Data/projects/pbt/allpbtfile/pbt_weights/checkpoint-2900",
                        help="Path to your trained T5 model directory")
    parser.add_argument("--duration", type=int, default=10,
                        help="Recording duration")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Audio sample rate")
    parser.add_argument("--save_audio", action="store_true",
                        help="Whether to save audio")
    parser.add_argument("--log_file", type=str,
                        default="trauma_inference_log.csv",
                        help="CSV log file path")
    args = parser.parse_args()

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    whisper_pipe = load_whisper(args.whisper_model, device)
    tokenizer, t5_model = load_t5_custom(args.t5_model_path, device)

    # Run inference
    run_all_inferences(whisper_pipe, tokenizer, t5_model, args, device)
