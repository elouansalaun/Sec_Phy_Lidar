from src.utils.config import load_config


def main():

    config = load_config()

    print("Project:", config["project"]["name"])

    print(
        "CRS:",
        config["spatial"]["crs"]
    )

    print(
        "LiDAR resolution:",
        config["spatial"]["resolution"]["lidar"]
    )


if __name__ == "__main__":
    main()