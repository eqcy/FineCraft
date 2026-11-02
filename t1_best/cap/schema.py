"""Keep the original eight T1 landmarks and add five cap-specific landmarks."""
CAMERAS = ('base_camera', 'hand_camera')
NAMES = ('body_lower', 'body_middle', 'body_upper', 'body_left', 'body_right',
         'tcp', 'left_pad', 'right_pad', 'cap_lower', 'cap_middle', 'cap_upper',
         'cap_left', 'cap_right')
K = len(NAMES)
KEYPOINT_DIM = 2 * K * 3
VISUAL_DIM = KEYPOINT_DIM + 2 * 3 * 128
