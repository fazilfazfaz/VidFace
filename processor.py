import argparse
import json
import multiprocessing
import os
import subprocess
import time
from os import mkdir
from shutil import rmtree
from threading import Thread

import face_recognition
import cv2
from math import floor, ceil

WORK_FOLDER = '.work'


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def get_frame_intervals(parts, frame_count):
    part_frame_count = frame_count / parts
    return [(floor(i * part_frame_count), ceil((i + 1) * part_frame_count)) for i in range(parts)]


def process_video_frames(file_path, search_face_encodings, frame_start, frame_end, face_presence):
    video_stream = cv2.VideoCapture(file_path)
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    current_frame_pos = frame_start
    frame_skip = 1
    while current_frame_pos <= frame_end:
        # print("Frame {} arrived in range {} to {}".format(current_frame_pos, frame_start, frame_end))
        # print("{} of range {} to {}".format((current_frame_pos - frame_start) / (frame_end - frame_start) * 100,
                                            # frame_start, frame_end))
        ret, frame = video_stream.read()
        if not ret:
            break
        current_frame_time = video_stream.get(cv2.CAP_PROP_POS_MSEC)
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = frame[:, :, ::-1]

        frame_face_locations = face_recognition.face_locations(rgb_frame, 1, 'cnn')
        frame_face_encodings = face_recognition.face_encodings(rgb_frame, frame_face_locations)

        face_found = False
        for frame_face_encoding in frame_face_encodings:
            matches = face_recognition.compare_faces(search_face_encodings, frame_face_encoding)
            if True in matches:
                face_found = True
                break

        face_presence[current_frame_time] = face_found
        if face_found:
            frame_skip = 5
        else:
            frame_skip = 10
        # if len(frame_face_encodings) > 0:
        #     if face_found_last is False:
        #         video_timings.append(video_stream.get(cv2.CAP_PROP_POS_MSEC))
        #         face_found_last = True
        # else:
        #     if face_found_last is True:
        #         video_timings.append(video_stream.get(cv2.CAP_PROP_POS_MSEC))
        #         face_found_last = False

        # for frame_face_encoding in frame_face_encodings:
        #     video_pos =
        # print("Found face at {}".format(video_stream.get(cv2.CAP_PROP_POS_MSEC)))
        # matches = face_recognition.compare_faces(search_face_encodings, frame_face_encoding)

        # print(1)

        current_frame_pos += frame_skip
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)

        # if len(video_timings) > 3:
        #     break


def main():
    parser = argparse.ArgumentParser(description='Strips out scenes from Videos based on provided faces')
    parser.add_argument('video_file')
    parser.add_argument('face_file')
    args = parser.parse_args()

    if not os.path.exists(args.video_file) or not os.path.exists(args.face_file):
        raise Exception("Invalid files passed")

    if os.path.exists(WORK_FOLDER):
        rmtree(WORK_FOLDER)
    os.mkdir(WORK_FOLDER)

    search_face_image = face_recognition.load_image_file(args.face_file)
    # face_locations = face_recognition.face_locations(search_face_image)
    search_face_encoding = face_recognition.face_encodings(search_face_image)[0]
    search_face_encodings = [search_face_encoding]

    video_file_name, video_file_extension = os.path.splitext(args.video_file)
    video_stream = cv2.VideoCapture(args.video_file)
    total_video_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    # video_stream.set(cv2.CAP_PROP_POS_FRAMES, 240)

    video_timings = []
    face_found_last = False

    no_of_threads = 3
    frame_intervals = get_frame_intervals(no_of_threads, total_video_frames)

    processes = []
    process_manager = multiprocessing.Manager()
    face_presences = process_manager.dict()

    processing_start_time = time.time()
    print("Processing the video")
    for frame_interval in frame_intervals:
        p = multiprocessing.Process(target=process_video_frames,
                                    args=(args.video_file, search_face_encodings, frame_interval[0], frame_interval[1],
                                          face_presences))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    print("Processing the video ended in {}".format(time.time() - processing_start_time))
    print("Processing the pieces")

    face_presence_timestamps = face_presences.keys()
    face_presence_timestamps.sort()

    face_last_found = False
    for frame_pos_time in face_presence_timestamps:
        if face_presences[frame_pos_time]:
            if not face_last_found:
                video_timings.append(frame_pos_time)
                face_last_found = True
        else:
            if face_last_found:
                video_timings.append(frame_pos_time)
                face_last_found = False

    print("Joining the pieces")

    video_files = []
    for index, part in enumerate(chunks(video_timings, 2)):
        if len(part) == 2:
            out_file = os.path.join(WORK_FOLDER, "p{}{}".format(index, video_file_extension))
            start = part[0]
            end = part[1] - part[0]
            start = start
            end = end
            part_command = 'ffmpeg -v quiet -y -ss {}ms -i "{}" -t {}ms -avoid_negative_ts 1 {}'.format(start,
                                                                                                            args.video_file,
                                                                                                            end,
                                                                                                            out_file)
            assert subprocess.call(part_command) == 0
            video_files.append(out_file)

    with open('files', 'w') as f:
        lines = map(lambda x: 'file \'{}\''.format(x), video_files)
        f.write("\n".join(lines))
    final_command = 'ffmpeg -v quiet -y -f concat -safe 0 -i {} -c copy {}'.format('files', video_file_name + '.stripped' + video_file_extension)
    subprocess.call(final_command)


if __name__ == '__main__':
    main()
