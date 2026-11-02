# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Track:
    '''Track class with state, covariance, id, score'''

    def __init__(self, meas, id):
        print('creating track no.', id)

        # rotation matrix from sensor to vehicle coordinates
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]

        ############
        # TODO Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        # Transform the measured object position from sensor coordinates
        # into vehicle coordinates using homogeneous coordinates.
        pos_sens = np.ones((4, 1))
        pos_sens[0:3] = meas.z[0:3]

        pos_veh = meas.sensor.sens_to_veh * pos_sens

        # Initialize the six-dimensional state vector:
        #
        # x = [px, py, pz, vx, vy, vz]^T
        #
        # A new measurement provides position information only.
        # Therefore, all initial velocity components are set to zero.
        self.x = np.matrix(
            np.zeros(
                (params.dim_state, 1)
            )
        )

        self.x[0:3] = pos_veh[0:3]
        self.x[3:6] = 0.0

        # Initialize the covariance matrix.
        self.P = np.matrix(
            np.zeros(
                (params.dim_state, params.dim_state)
            )
        )

        # Transform the measurement covariance from sensor coordinates
        # into vehicle coordinates.
        self.P[0:3, 0:3] = (
            M_rot
            * meas.R
            * M_rot.T
        )

        # A new track has no direct velocity measurement.
        # Large variances are therefore used for the velocity components.
        self.P[3, 3] = params.sigma_p44 ** 2
        self.P[4, 4] = params.sigma_p55 ** 2
        self.P[5, 5] = params.sigma_p66 ** 2

        # New tracks start in the initialized state.
        self.state = 'initialized'

        # A new track starts with one successful detection within
        # the score-management window.
        self.score = 1.0 / params.window

        ############
        # END student code
        ############

        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height

        # transform rotation from sensor to vehicle coordinates
        yaw_argument = (
            M_rot[0, 0] * np.cos(meas.yaw)
            + M_rot[0, 1] * np.sin(meas.yaw)
        )

        # Numerical errors can occasionally push the argument slightly
        # outside the valid range of arccos.
        yaw_argument = np.clip(
            float(yaw_argument),
            -1.0,
            1.0
        )

        self.yaw = np.arccos(
            yaw_argument
        )

        self.t = meas.t

    def set_x(self, x):
        self.x = x

    def set_P(self, P):
        self.P = P

    def set_t(self, t):
        self.t = t

    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim

            self.width = (
                c * meas.width
                + (1 - c) * self.width
            )

            self.length = (
                c * meas.length
                + (1 - c) * self.length
            )

            self.height = (
                c * meas.height
                + (1 - c) * self.height
            )

            M_rot = meas.sensor.sens_to_veh

            yaw_argument = (
                M_rot[0, 0] * np.cos(meas.yaw)
                + M_rot[0, 1] * np.sin(meas.yaw)
            )

            # Prevent numerical errors in arccos.
            yaw_argument = np.clip(
                float(yaw_argument),
                -1.0,
                1.0
            )

            # transform rotation from sensor to vehicle coordinates
            measured_yaw = np.arccos(
                yaw_argument
            )

            self.yaw = (
                c * measured_yaw
                + (1 - c) * self.yaw
            )


###################


class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''

    def __init__(self):
        self.N = 0  # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []

    def manage_tracks(
        self,
        unassigned_tracks,
        unassigned_meas,
        meas_list
    ):
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############

        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]

            # check visibility
            if meas_list:  # if not empty
                sensor = meas_list[0].sensor

                if sensor.in_fov(track.x):
                    track.score = max(
                        0.0,
                        track.score - 1.0 / params.window
                    )

        # delete old tracks
        tracks_to_delete = []

        for track in self.track_list:

            # Delete a confirmed track when its score falls below
            # the configured deletion threshold.
            delete_confirmed = (
                track.state == 'confirmed'
                and track.score < params.delete_threshold
            )

            # An initialized or tentative track is removed when its
            # score reaches zero.
            delete_unconfirmed = (
                track.state != 'confirmed'
                and track.score <= 0.0
            )

            # Delete a track when its positional uncertainty becomes
            # too large.
            delete_covariance = (
                float(track.P[0, 0]) > params.max_P
                or float(track.P[1, 1]) > params.max_P
            )

            if (
                delete_confirmed
                or delete_unconfirmed
                or delete_covariance
            ):
                tracks_to_delete.append(track)

        # Tracks are collected first and deleted afterwards to avoid
        # modifying track_list while iterating over it.
        for track in tracks_to_delete:
            self.delete_track(track)

        ############
        # END student code
        ############

        # initialize new track with unassigned measurement
        for j in unassigned_meas:
            if meas_list[j].sensor.name == 'lidar':
                # only initialize with lidar measurements
                self.init_track(
                    meas_list[j]
                )

    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(
            meas,
            self.last_id + 1
        )

        self.addTrackToList(track)

    def delete_track(self, track):
        print(
            'deleting track no.',
            track.id
        )

        self.track_list.remove(track)

        # Keep the current number of tracks synchronized with track_list.
        self.N = len(self.track_list)

    def handle_updated_track(self, track):
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        # Increase the track score after a successful measurement
        # association, while keeping it within the interval [0, 1].
        track.score = min(
            1.0,
            track.score + 1.0 / params.window
        )

        # Promote initialized tracks to tentative once they have
        # received another successful measurement update.
        if (
            track.state == 'initialized'
            and track.score > 1.0 / params.window
        ):
            track.state = 'tentative'

        # Promote a track to confirmed once its score exceeds
        # the configured confirmation threshold.
        if track.score >= params.confirmed_threshold:
            track.state = 'confirmed'

        ############
        # END student code
        ############