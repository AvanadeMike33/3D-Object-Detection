# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(
        os.path.join(
            os.getcwd(),
            os.path.expanduser(__file__)
        )
    )
)

sys.path.append(
    os.path.normpath(
        os.path.join(
            SCRIPT_DIR,
            PACKAGE_PARENT
        )
    )
)

import misc.params as params


class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''

    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):

        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        # Initialize the association matrix with infinity.
        #
        # Rows correspond to tracks.
        # Columns correspond to measurements.
        #
        # An infinite value represents a track-measurement pair that
        # does not pass the statistical gating test.
        self.association_matrix = np.matrix(
            np.inf * np.ones(
                (
                    len(track_list),
                    len(meas_list)
                )
            )
        )

        # At the beginning, all tracks and measurements are unassigned.
        self.unassigned_tracks = list(
            range(len(track_list))
        )

        self.unassigned_meas = list(
            range(len(meas_list))
        )

        # Calculate the Mahalanobis distance for every possible
        # track-measurement pair.
        best_camera_mhd = np.inf

        for i, track in enumerate(track_list):

            for j, meas in enumerate(meas_list):

                # Only consider objects that lie inside the field of view
                # of the sensor that produced the measurement.
                if not meas.sensor.in_fov(track.x):
                    continue

                # Calculate Mahalanobis distance.
                MHD = self.MHD(
                    track,
                    meas,
                    KF
                )
                
                if meas.sensor.name == 'camera':
                    best_camera_mhd = min(
                        best_camera_mhd,
                        float(MHD)
                    )
                # Store only distances that pass the chi-square gate.
                if self.gating(
                    MHD,
                    meas.sensor
                ):
                    self.association_matrix[i, j] = MHD

        # Temporary camera association diagnostic.
        # This prints only one summary per camera-association cycle,
        # instead of one line for every track-measurement combination.
        if (
            len(meas_list) > 0
            and meas_list[0].sensor.name == 'camera'
        ):
            association_array = np.asarray(
                self.association_matrix
            )

            finite_values = association_array[
                np.isfinite(association_array)
            ]

            print(
                "CAMERA ASSOCIATIONS PASSING GATE:",
                len(finite_values)
            )

            if len(finite_values) > 0:
                print(
                    "BEST CAMERA MHD:",
                    float(np.min(finite_values))
                )

        if (
            len(meas_list) > 0
            and meas_list[0].sensor.name == 'camera'
        ):
            print(
                "BEST RAW CAMERA MHD:",
                best_camera_mhd
            )

            print(
                "CAMERA GATE:",
                chi2.ppf(
                    params.gating_threshold,
                    meas_list[0].sensor.dim_meas
                )
            )   
                
        ############
        # END student code
        ############

    def get_closest_track_and_meas(self):

        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # No association is possible when one dimension of the
        # association matrix is empty.
        if (
            self.association_matrix.shape[0] == 0
            or self.association_matrix.shape[1] == 0
        ):
            return np.nan, np.nan

        # Find minimum entry in association matrix.
        min_value = np.min(
            self.association_matrix
        )

        # If the minimum is infinity, none of the remaining
        # track-measurement combinations passed the gate.
        if np.isinf(min_value):
            return np.nan, np.nan

        # Determine the row and column of the minimum entry.
        flat_index = np.argmin(
            self.association_matrix
        )

        ind_track, ind_meas = np.unravel_index(
            flat_index,
            self.association_matrix.shape
        )

        # Convert matrix indices to the original track and
        # measurement indices.
        update_track = self.unassigned_tracks[
            ind_track
        ]

        update_meas = self.unassigned_meas[
            ind_meas
        ]

        # Delete the selected track row.
        self.association_matrix = np.delete(
            self.association_matrix,
            ind_track,
            axis=0
        )

        # Delete the selected measurement column.
        self.association_matrix = np.delete(
            self.association_matrix,
            ind_meas,
            axis=1
        )

        # Remove selected track and measurement from the
        # unassigned lists.
        self.unassigned_tracks.pop(
            ind_track
        )

        self.unassigned_meas.pop(
            ind_meas
        )

        ############
        # END student code
        ############

        return update_track, update_meas

    def gating(self, MHD, sensor):

        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############

        # Calculate the chi-square threshold according to the
        # configured gating probability and measurement dimension.
        limit = chi2.ppf(
            params.gating_threshold,
            sensor.dim_meas
        )

        return float(MHD) < limit

        ############
        # END student code
        ############

    def MHD(self, track, meas, KF):

        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############

        # Calculate measurement matrix / camera Jacobian.
        H = meas.sensor.get_H(
            track.x
        )

        # Calculate measurement residual:
        #
        # gamma = z - h(x)
        gamma = KF.gamma(
            track,
            meas
        )

        # Calculate residual covariance:
        #
        # S = H * P * H^T + R
        S = KF.S(
            track,
            meas,
            H
        )

        # Calculate squared Mahalanobis distance:
        #
        # MHD = gamma^T * S^-1 * gamma
        #
        # Solving S*x = gamma is numerically preferable to
        # explicitly computing the inverse of S.
        MHD = gamma.T * np.linalg.solve(
            S,
            gamma
        )

        return float(
            np.asarray(MHD).squeeze()
        )

        ############
        # END student code
        ############

    def associate_and_update(self, manager, meas_list, KF):

        # associate measurements and tracks
        self.associate(
            manager.track_list,
            meas_list,
            KF
        )

        # update associated tracks with measurements
        while (
            self.association_matrix.shape[0] > 0
            and self.association_matrix.shape[1] > 0
        ):

            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()

            if np.isnan(ind_track):
                print('---no more associations---')
                break

            track = manager.track_list[
                int(ind_track)
            ]

            measurement = meas_list[
                int(ind_meas)
            ]

            # check visibility, only update tracks in fov
            if not measurement.sensor.in_fov(
                track.x
            ):
                continue

            # Kalman update
            print(
                'update track',
                track.id,
                'with',
                measurement.sensor.name,
                'measurement',
                int(ind_meas)
            )

            KF.update(
                track,
                measurement
            )

            # Update the timestamp of the track using the timestamp
            # of the associated measurement.
            track.set_t(
                measurement.t
            )

            # update score and track state
            manager.handle_updated_track(
                track
            )

            # save updated track
            manager.track_list[
                int(ind_track)
            ] = track

        # run track management
        manager.manage_tracks(
            self.unassigned_tracks,self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print( 'track',track.id,'score =',track.score)