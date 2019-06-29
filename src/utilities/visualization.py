def adjust_spines(ax, spines, offset=3, smart_bounds=False):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', offset))
            spine.set_smart_bounds(smart_bounds)
        else:
            spine.set_color('None')  # don't draw spine

    # Turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # No y-axis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # No x-axis ticks
        ax.xaxis.set_ticks([])
