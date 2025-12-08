from enum import Enum


class TeleportMode(Enum):
    PROPORTIONAL = "proportional"
    TOP_K = "top_k"


class PrizeAllocation(Enum):
    LINEAR = "linear"
    EQUAL = "equal"
    EXPONENTIAL = "exponential"


def get_retrieval_func(retrieval_method: str, tele_mode: str = None, pcst: bool = False, prize_allocation: str = None):
    """
    Get the appropriate retrieval function based on the method and parameters.

    Args:
        retrieval_method: One of 'pcst', 'k_hop', or 'ppr'
        tele_mode: Teleport mode for PPR retrieval. One of 'proportional', 'top_k'.
                   Only used when retrieval_method='ppr'. Must be None for other methods.
        pcst: Whether to use PCST mode for PPR retrieval. Only used when retrieval_method='ppr'.
        prize_allocation: Prize allocation mode. One of 'linear', 'equal', 'exponential'.
                         Used when retrieval_method='pcst' or when retrieval_method='ppr' with tele_mode='top_k' or pcst=True.
                         Must be None when retrieval_method='k_hop' or when retrieval_method='ppr' with tele_mode='proportional' and pcst=False.
                         Defaults to 'linear' if not specified when required.

    Returns:
        The retrieval function to use for subgraph extraction.
    
    Raises:
        ValueError: If parameters are not set correctly for the given retrieval_method.
    """
    # Validate parameters based on retrieval_method
    if retrieval_method == 'k_hop':
        if tele_mode is not None:
            raise ValueError(f"tele_mode must be None when retrieval_method='k_hop', but got: {tele_mode}")
        if prize_allocation is not None:
            raise ValueError(f"prize_allocation must be None when retrieval_method='k_hop', but got: {prize_allocation}")
    elif retrieval_method == 'pcst':
        if tele_mode is not None:
            raise ValueError(f"tele_mode must be None when retrieval_method='pcst', but got: {tele_mode}")
    
    if retrieval_method == 'pcst':
        from src.dataset.utils.retrieval import retrieval_via_pcst_fn
        
        # Map string to enum for prize_allocation
        if prize_allocation is None:
            prize_allocation_enum = PrizeAllocation.LINEAR
        else:
            prize_allocation_map = {
                'linear': PrizeAllocation.LINEAR,
                'equal': PrizeAllocation.EQUAL,
                'exponential': PrizeAllocation.EXPONENTIAL,
            }
            if prize_allocation not in prize_allocation_map:
                raise ValueError(f"Unknown prize_allocation: {prize_allocation}. Must be one of: {list(prize_allocation_map.keys())}")
            prize_allocation_enum = prize_allocation_map[prize_allocation]
        
        return retrieval_via_pcst_fn(prize_allocation_enum)
    elif retrieval_method == 'k_hop':
        from src.dataset.utils.k_hop import retrieval_via_k_hop as retrieval_func
        return retrieval_func
    elif retrieval_method == 'ppr':
        from src.dataset.utils.personalized_pagerank import retrieval_via_pagerank_fn

        # Map string to enum for tele_mode
        if tele_mode is None:
            tele_mode_enum = TeleportMode.PROPORTIONAL
        else:
            tele_mode_map = {
                'proportional': TeleportMode.PROPORTIONAL,
                'top_k': TeleportMode.TOP_K,
            }
            if tele_mode not in tele_mode_map:
                raise ValueError(f"Unknown tele_mode: {tele_mode}. Must be one of: {list(tele_mode_map.keys())}")
            tele_mode_enum = tele_mode_map[tele_mode]

        # Map string to enum for prize_allocation
        # Used when tele_mode is TOP_K (for personalization vector) or when pcst=True (for PCST prize allocation)
        if tele_mode_enum == TeleportMode.TOP_K or pcst:
            if prize_allocation is None:
                prize_allocation_enum = PrizeAllocation.LINEAR
            else:
                prize_allocation_map = {
                    'linear': PrizeAllocation.LINEAR,
                    'equal': PrizeAllocation.EQUAL,
                    'exponential': PrizeAllocation.EXPONENTIAL,
                }
                if prize_allocation not in prize_allocation_map:
                    raise ValueError(f"Unknown prize_allocation: {prize_allocation}. Must be one of: {list(prize_allocation_map.keys())}")
                prize_allocation_enum = prize_allocation_map[prize_allocation]
        else:
            # If tele_mode is PROPORTIONAL and pcst is False, prize_allocation must be None
            if prize_allocation is not None:
                raise ValueError(f"prize_allocation must be None when retrieval_method='ppr' with tele_mode='proportional' and pcst=False, but got: {prize_allocation}")
            prize_allocation_enum = None

        return retrieval_via_pagerank_fn(pcst, tele_mode_enum, prize_allocation_enum)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}. Must be one of: 'pcst', 'k_hop', 'ppr'")

