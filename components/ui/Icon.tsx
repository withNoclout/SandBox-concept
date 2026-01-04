import React from 'react';
import { Icon as IconifyIcon } from '@iconify/react';

interface IconProps extends React.HTMLAttributes<HTMLSpanElement> {
    icon: string;
    width?: string | number;
    height?: string | number;
}

export const Icon = ({ icon, width, height, className, ...props }: IconProps) => {
    return (
        <span className={`inline-flex items-center justify-center ${className || ''}`} {...props}>
            <IconifyIcon icon={icon} width={width} height={height} />
        </span>
    );
};
