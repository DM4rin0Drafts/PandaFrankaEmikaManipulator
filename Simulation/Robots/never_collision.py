# Copied from ROS
# Links that cannot collide or just can ignore if they collide
PANDA_NEVER_COLLISIONS = [
    ('panda_hand', 'panda_leftfinger'), ('panda_hand', 'panda_rightfinger'),          # Adjecent Links
    ('panda_link1', 'panda_link2'), ('panda_link2', 'panda_link3'), ('panda_link3', 'panda_link4'),
    ('panda_link4', 'panda_link5'), ('panda_link5', 'panda_link6'), ('panda_link6', 'panda_link7'),
    ('panda_link7', 'panda_hand'),
    ('panda_leftfinger', 'panda_rightfinger'), ('panda_link5', 'panda_hand'), ('panda_link5', 'panda_link7'),    # Collision by Default (when spawn to simulation)
    # ('panda_link0', 'panda_link2'), ('panda_link0', 'panda_link3'), ('panda_link0', 'panda_link4'), # NEVER COLLISIONS
    ('panda_link1', 'panda_link3'), ('panda_link1', 'panda_link4'), ('panda_link2', 'panda_link4'),
    ('panda_link3', 'panda_hand'), ('panda_link3', 'panda_leftfinger'), ('panda_link3', 'panda_link5'),
    ('panda_link3', 'panda_link6'), ('panda_link3', 'panda_link7'), ('panda_link3', 'panda_rightfinger'),
    ('panda_link4', 'panda_hand'), ('panda_link4', 'panda_leftfinger'), ('panda_link4', 'panda_link6'),
    ('panda_link4', 'panda_link7'), ('panda_link4', 'panda_rightfinger'), ('panda_link6', 'panda_hand'),
    ('panda_link6', 'panda_leftfinger'), ('panda_link6', 'panda_rightfinger'),
    ('panda_link7', 'panda_leftfinger'), ('panda_link7', 'panda_rightfinger')
]
