from setuptools import setup

package_name = "yolo_obstacle_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=["obstacle_detector"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "httpx"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "detector_node = obstacle_detector.hard_soft_obstaclenode:main",
            "overseer_node = obstacle_detector.overseer_node:main",
            "gpt_query_node = obstacle_detector.gpt_query_node:main",
            "hazard_detector_node = obstacle_detector.hazard_detector_node:main",
            "qr_detector_node = obstacle_detector.qr_detector_node:main",
            "webcam_test_node = obstacle_detector.webcam_test_node:main",
            "detector_visualizer_node = obstacle_detector.detector_visualizer_node:main",
            "windows_detector_test = obstacle_detector.windows_detector_test:main",
            "semantic_webcam_test = obstacle_detector.semantic_webcam_test:main",
        ],
    },
)
