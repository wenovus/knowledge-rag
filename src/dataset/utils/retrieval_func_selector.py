def get_retrieval_func(retrieval_method: str, tele_mode: str = None, pcst: bool = False):
    """
    Get the appropriate retrieval function based on the method and parameters.

    Args:
        retrieval_method: One of 'pcst', 'k_hop', or 'ppr'
        tele_mode: Teleport mode for PPR retrieval. One of 'proportional', 'top_k_linear',
                   'top_k_equal', 'top_k_exponential'. Only used when retrieval_method='ppr'.
        pcst: Whether to use PCST mode for PPR retrieval. Only used when retrieval_method='ppr'.

    Returns:
        The retrieval function to use for subgraph extraction.
    """
    if retrieval_method == 'pcst':
        from src.dataset.utils.retrieval import retrieval_via_pcst as retrieval_func
        return retrieval_func
    elif retrieval_method == 'k_hop':
        from src.dataset.utils.k_hop import retrieval_via_k_hop as retrieval_func
        return retrieval_func
    elif retrieval_method == 'ppr':
        from src.dataset.utils.personalized_pagerank import retrieval_via_pagerank_fn, TeleportMode

        # Map string to enum
        if tele_mode is None:
            tele_mode_enum = TeleportMode.PROPORTIONAL
        else:
            tele_mode_map = {
                'proportional': TeleportMode.PROPORTIONAL,
                'top_k_linear': TeleportMode.TOP_K_LINEAR,
                'top_k_equal': TeleportMode.TOP_K_EQUAL,
                'top_k_exponential': TeleportMode.TOP_K_EXPONENTIAL,
            }
            if tele_mode not in tele_mode_map:
                raise ValueError(f"Unknown tele_mode: {tele_mode}. Must be one of: {list(tele_mode_map.keys())}")
            tele_mode_enum = tele_mode_map[tele_mode]

        return retrieval_via_pagerank_fn(pcst, tele_mode_enum)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}. Must be one of: 'pcst', 'k_hop', 'ppr'")

