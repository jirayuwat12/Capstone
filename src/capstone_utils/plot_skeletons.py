def plot_skeletons_video(
    joints: list,
    file_path: str,
    video_name: str,
    references: list | None = None,
    skip_frames: int = 1,
    sequence_ID: str | None = None,
    pad_token: int = 0,
    frame_offset: tuple[int, int] = (0, 0),
    debug: bool = False,
) -> None:
    """
    DEPRECATED: Please use `create_sign_language_video` from `capstone_utils.plot_all_body` module instead.
    """
    raise DeprecationWarning(
        "This function is deprecated. Please use `create_sign_language_video` from `capstone_utils.plot_all_body` module instead."
    )
