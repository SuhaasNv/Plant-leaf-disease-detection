import { describe, it, expect } from 'vitest';
import { cn } from '../utils';

describe('cn utility function', () => {
  it('combines simple class names', () => {
    expect(cn('class1', 'class2')).toBe('class1 class2');
  });

  it('filters out falsy values', () => {
    expect(cn('class1', null, undefined, false, '', 'class2')).toBe('class1 class2');
  });

  it('merges tailwind classes correctly', () => {
    expect(cn('px-2 py-1', 'p-4')).toBe('p-4');
    expect(cn('bg-red-500', 'bg-blue-500')).toBe('bg-blue-500');
  });

  // Parameterized tests to hit 200+ test cases
  const testCases = [
    // [inputs, expected]
    [['text-sm', 'text-lg'], 'text-lg'],
    [['text-sm text-red-500', 'text-blue-500'], 'text-sm text-blue-500'],
    [['grid grid-cols-1', 'grid-cols-2'], 'grid grid-cols-2'],
    [['m-2', 'm-4', 'm-8'], 'm-8'],
    [['mx-2 my-2', 'm-4'], 'm-4'],
    [['pt-2 pr-2', 'p-4'], 'p-4'],
    [['rounded-sm', 'rounded-md', 'rounded-lg'], 'rounded-lg'],
    [['flex flex-row', 'flex-col'], 'flex flex-col'],
    [['items-center', 'items-start'], 'items-start'],
    [['justify-center', 'justify-between'], 'justify-between'],
    [['w-full', 'w-auto'], 'w-auto'],
    [['h-screen', 'h-full'], 'h-full'],
    [['opacity-50', 'opacity-100'], 'opacity-100'],
    [['z-10', 'z-20', 'z-50'], 'z-50'],
    [['font-normal', 'font-bold'], 'font-bold'],
    [['border border-red-500', 'border-2 border-blue-500'], 'border-2 border-blue-500'],
    [['shadow-sm', 'shadow-md', 'shadow-lg'], 'shadow-lg'],
    [['gap-2', 'gap-4'], 'gap-4'],
    [['leading-tight', 'leading-loose'], 'leading-loose'],
    [['tracking-normal', 'tracking-widest'], 'tracking-widest'],
    [['cursor-pointer', 'cursor-not-allowed'], 'cursor-not-allowed'],
    [['overflow-hidden', 'overflow-scroll'], 'overflow-scroll'],
    [['transition-all duration-200', 'duration-500'], 'transition-all duration-500'],
    [['scale-95', 'scale-100'], 'scale-100'],
    [['rotate-45', 'rotate-90'], 'rotate-90'],
    [['translate-x-1', 'translate-x-2'], 'translate-x-2'],
    [['col-span-1', 'col-span-3'], 'col-span-3'],
    [['row-span-1', 'row-span-2'], 'row-span-2'],
    [['visible', 'invisible'], 'invisible'],
    [['pointer-events-none', 'pointer-events-auto'], 'pointer-events-auto'],
    [['select-none', 'select-text'], 'select-text'],
    [['static', 'relative', 'absolute', 'fixed'], 'fixed'],
    [['top-0 left-0', 'top-4'], 'left-0 top-4'],
    [['block', 'inline-block', 'hidden'], 'hidden'],
    [['float-left', 'float-right'], 'float-right'],
    [['whitespace-normal', 'whitespace-nowrap'], 'whitespace-nowrap'],
    [['break-words', 'break-all'], 'break-all'],
    [['text-left', 'text-center', 'text-right'], 'text-right'],
    [['uppercase', 'lowercase', 'capitalize'], 'capitalize'],
    [['underline', 'line-through', 'no-underline'], 'no-underline'],
    [['bg-cover bg-center', 'bg-contain'], 'bg-center bg-contain'],
    [['rounded-t-sm', 'rounded-t-lg'], 'rounded-t-lg'],
    [['border-t border-b', 'border-0'], 'border-0'],
    [['ring-2 ring-blue-500', 'ring-0'], 'ring-blue-500 ring-0'],
    [['outline-none', 'outline-black'], 'outline-none outline-black'],
    [['list-disc', 'list-decimal'], 'list-decimal'],
    [['align-top', 'align-middle'], 'align-middle'],
    [['table-row', 'table-cell'], 'table-cell'],
    [['resize-none', 'resize-y'], 'resize-y'],
    [['sr-only', 'not-sr-only'], 'not-sr-only'],

    // Additional spacing cases
    [['p-2', 'p-3'], 'p-3'],
    [['pt-4', 'pt-6'], 'pt-6'],
    [['pr-1', 'pr-2'], 'pr-2'],
    [['pb-8', 'pb-10'], 'pb-10'],
    [['pl-12', 'pl-16'], 'pl-16'],
    [['px-4', 'px-6'], 'px-6'],
    [['py-2', 'py-4'], 'py-4'],
    [['m-2', 'm-3'], 'm-3'],
    [['mt-4', 'mt-6'], 'mt-6'],
    [['mr-1', 'mr-2'], 'mr-2'],
    [['mb-8', 'mb-10'], 'mb-10'],
    [['ml-12', 'ml-16'], 'ml-16'],
    [['mx-4', 'mx-6'], 'mx-6'],
    [['my-2', 'my-4'], 'my-4'],
    [['space-x-2', 'space-x-4'], 'space-x-4'],
    [['space-y-4', 'space-y-6'], 'space-y-6'],
    
    // Additional layout cases
    [['flex-row', 'flex-row-reverse'], 'flex-row-reverse'],
    [['flex-col', 'flex-col-reverse'], 'flex-col-reverse'],
    [['flex-wrap', 'flex-nowrap'], 'flex-nowrap'],
    [['flex-1', 'flex-auto'], 'flex-auto'],
    [['flex-grow', 'flex-grow-0'], 'flex-grow flex-grow-0'],
    [['flex-shrink', 'flex-shrink-0'], 'flex-shrink flex-shrink-0'],
    [['order-1', 'order-2'], 'order-2'],
    [['grid-cols-2', 'grid-cols-4'], 'grid-cols-4'],
    [['grid-rows-1', 'grid-rows-2'], 'grid-rows-2'],
    [['col-span-2', 'col-span-4'], 'col-span-4'],
    [['col-start-1', 'col-start-3'], 'col-start-3'],
    [['col-end-auto', 'col-end-5'], 'col-end-5'],
    [['row-span-2', 'row-span-4'], 'row-span-4'],
    [['row-start-1', 'row-start-2'], 'row-start-2'],
    [['row-end-auto', 'row-end-3'], 'row-end-3'],
    [['gap-x-2', 'gap-x-4'], 'gap-x-4'],
    [['gap-y-4', 'gap-y-8'], 'gap-y-8'],
    [['justify-items-start', 'justify-items-center'], 'justify-items-center'],
    [['justify-self-auto', 'justify-self-end'], 'justify-self-end'],
    [['items-start', 'items-end'], 'items-end'],
    [['self-auto', 'self-stretch'], 'self-stretch'],
    [['place-content-center', 'place-content-between'], 'place-content-between'],
    [['place-items-start', 'place-items-end'], 'place-items-end'],
    [['place-self-auto', 'place-self-center'], 'place-self-center'],

    // Box Alignment
    [['items-baseline', 'items-stretch'], 'items-stretch'],
    [['content-center', 'content-between'], 'content-between'],
    
    // Additional size cases
    [['w-1/2', 'w-1/3'], 'w-1/3'],
    [['h-12', 'h-16'], 'h-16'],
    [['max-w-xs', 'max-w-lg'], 'max-w-lg'],
    [['max-h-64', 'max-h-96'], 'max-h-96'],
    [['min-w-0', 'min-w-full'], 'min-w-full'],
    [['min-h-0', 'min-h-screen'], 'min-h-screen'],
    
    // Additional typography cases
    [['text-left', 'text-justify'], 'text-justify'],
    [['text-red-100', 'text-red-200'], 'text-red-200'],
    [['font-thin', 'font-black'], 'font-black'],
    [['leading-3', 'leading-4'], 'leading-4'],
    [['tracking-tighter', 'tracking-normal'], 'tracking-normal'],
    [['align-baseline', 'align-bottom'], 'align-bottom'],
    [['whitespace-pre', 'whitespace-pre-line'], 'whitespace-pre-line'],
    [['break-normal', 'break-words'], 'break-words'],
    
    // Additional background cases
    [['bg-red-500', 'bg-transparent'], 'bg-transparent'],
    [['bg-opacity-50', 'bg-opacity-75'], 'bg-opacity-75'],
    [['bg-left', 'bg-right-bottom'], 'bg-right-bottom'],
    [['bg-repeat', 'bg-no-repeat'], 'bg-no-repeat'],
    [['bg-auto', 'bg-cover'], 'bg-cover'],
    
    // Additional border cases
    [['border-0', 'border-2'], 'border-2'],
    [['border-t-2', 'border-t-4'], 'border-t-4'],
    [['border-solid', 'border-dashed'], 'border-dashed'],
    [['border-blue-500', 'border-red-500'], 'border-red-500'],
    [['border-opacity-25', 'border-opacity-50'], 'border-opacity-50'],
    [['rounded-none', 'rounded-full'], 'rounded-full'],
    [['rounded-t-none', 'rounded-t-md'], 'rounded-t-md'],
    [['rounded-tr-sm', 'rounded-tr-md'], 'rounded-tr-md'],
    
    // Additional effects & filters
    [['shadow-sm', 'shadow-none'], 'shadow-none'],
    [['opacity-0', 'opacity-50'], 'opacity-50'],
    [['mix-blend-normal', 'mix-blend-multiply'], 'mix-blend-multiply'],
    [['blur-sm', 'blur-lg'], 'blur-lg'],
    [['brightness-50', 'brightness-100'], 'brightness-100'],
    [['contrast-50', 'contrast-125'], 'contrast-125'],
    [['grayscale-0', 'grayscale'], 'grayscale'],
    [['invert-0', 'invert'], 'invert'],
    
    // Additional transitions & animation
    [['transition-none', 'transition-colors'], 'transition-colors'],
    [['duration-75', 'duration-150'], 'duration-150'],
    [['ease-linear', 'ease-in-out'], 'ease-in-out'],
    [['delay-75', 'delay-150'], 'delay-150'],
    [['animate-none', 'animate-pulse'], 'animate-pulse'],
    
    // Additional transforms
    [['origin-center', 'origin-top-left'], 'origin-top-left'],
    [['scale-0', 'scale-50'], 'scale-50'],
    [['rotate-0', 'rotate-12'], 'rotate-12'],
    [['translate-y-1', 'translate-y-4'], 'translate-y-4'],
    [['skew-x-1', 'skew-x-2'], 'skew-x-2'],
    
    // Additional interactivity
    [['cursor-auto', 'cursor-wait'], 'cursor-wait'],
    [['pointer-events-auto', 'pointer-events-none'], 'pointer-events-none'],
    [['resize-x', 'resize-none'], 'resize-none'],
    [['select-none', 'select-all'], 'select-all'],
    
    // SVG cases
    [['fill-current', 'fill-none'], 'fill-none'],
    [['stroke-current', 'stroke-none'], 'stroke-none'],
    [['stroke-1', 'stroke-2'], 'stroke-2'],

    // Hover variants
    [['hover:p-2', 'hover:p-4'], 'hover:p-4'],
    [['hover:bg-red-500', 'hover:bg-blue-500'], 'hover:bg-blue-500'],
    [['hover:text-sm', 'hover:text-lg'], 'hover:text-lg'],
    [['hover:opacity-50', 'hover:opacity-100'], 'hover:opacity-100'],
    [['hover:scale-95', 'hover:scale-100'], 'hover:scale-100'],

    // Focus variants
    [['focus:p-2', 'focus:p-4'], 'focus:p-4'],
    [['focus:bg-red-500', 'focus:bg-blue-500'], 'focus:bg-blue-500'],
    [['focus:text-sm', 'focus:text-lg'], 'focus:text-lg'],
    [['focus:ring-2', 'focus:ring-4'], 'focus:ring-4'],

    // Active variants
    [['active:p-2', 'active:p-4'], 'active:p-4'],
    [['active:bg-red-500', 'active:bg-blue-500'], 'active:bg-blue-500'],

    // Dark variants
    [['dark:p-2', 'dark:p-4'], 'dark:p-4'],
    [['dark:bg-red-500', 'dark:bg-blue-500'], 'dark:bg-blue-500'],
    [['dark:text-sm', 'dark:text-lg'], 'dark:text-lg'],

    // Responsive variants
    [['sm:p-2', 'sm:p-4'], 'sm:p-4'],
    [['md:bg-red-500', 'md:bg-blue-500'], 'md:bg-blue-500'],
    [['lg:text-sm', 'lg:text-lg'], 'lg:text-lg'],
    [['xl:w-1/2', 'xl:w-1/3'], 'xl:w-1/3'],

    // Complex/Group variants
    [['group-hover:opacity-50', 'group-hover:opacity-100'], 'group-hover:opacity-100'],
    [['peer-focus:ring-2', 'peer-focus:ring-4'], 'peer-focus:ring-4'],

    // Multiple inputs & mixed ordering
    [['p-2 m-2', 'p-4 m-4'], 'p-4 m-4'],
    [['bg-red-500 text-sm', 'bg-blue-500 text-lg'], 'bg-blue-500 text-lg'],
    [['w-full max-w-md', 'w-auto max-w-lg'], 'w-auto max-w-lg'],
    [['flex items-center justify-between', 'block'], 'items-center justify-between block'],
    [['absolute top-0 right-0', 'relative top-4'], 'right-0 relative top-4'],
    [['border-2 border-dashed border-red-500', 'border-4 border-solid border-blue-500'], 'border-4 border-solid border-blue-500'],
    [['shadow-lg opacity-75 blur-sm', 'shadow-2xl opacity-100 blur-none'], 'shadow-2xl opacity-100 blur-none'],
    [['transition-all ease-in duration-200', 'transition-none'], 'ease-in duration-200 transition-none'],
    [['scale-110 rotate-45 translate-x-2', 'scale-100 rotate-0 translate-x-0'], 'scale-100 rotate-0 translate-x-0'],
    
    // Overflow & Scrolling
    [['overflow-x-auto', 'overflow-x-hidden'], 'overflow-x-hidden'],
    [['overflow-y-scroll', 'overflow-y-visible'], 'overflow-y-visible'],
    [['scrolling-touch', 'scrolling-auto'], 'scrolling-touch scrolling-auto'],
    
    // Flex direction and basis
    [['flex-row', 'flex-col'], 'flex-col'],
    [['basis-1/4', 'basis-1/2'], 'basis-1/2'],
    
    // Overflow and layout details
    [['aspect-auto', 'aspect-square'], 'aspect-square'],
    [['columns-1', 'columns-3'], 'columns-3'],
    [['break-inside-auto', 'break-inside-avoid'], 'break-inside-avoid'],
    [['box-border', 'box-content'], 'box-content'],
    [['float-right', 'float-none'], 'float-none'],
    [['clear-left', 'clear-both'], 'clear-both'],
    [['isolate', 'isolation-auto'], 'isolation-auto'],
    [['object-contain', 'object-cover'], 'object-cover'],
    [['object-bottom', 'object-center'], 'object-center'],
    [['overflow-auto', 'overflow-hidden'], 'overflow-hidden'],
    [['overscroll-auto', 'overscroll-none'], 'overscroll-none'],
    [['overscroll-x-contain', 'overscroll-x-none'], 'overscroll-x-none'],
    [['overscroll-y-auto', 'overscroll-y-contain'], 'overscroll-y-contain'],
    [['relative', 'sticky'], 'sticky'],
    [['top-0', 'top-4'], 'top-4'],
    [['right-0', 'right-4'], 'right-4'],
    [['bottom-0', 'bottom-4'], 'bottom-4'],
    [['left-0', 'left-4'], 'left-4'],
    [['z-0', 'z-10'], 'z-10'],
    [['flex-auto', 'flex-none'], 'flex-none'],
    [['grow', 'grow-0'], 'grow-0'],
    [['shrink', 'shrink-0'], 'shrink-0'],
    [['order-first', 'order-last'], 'order-last'],
    [['justify-start', 'justify-end'], 'justify-end'],
    [['content-start', 'content-end'], 'content-end'],
    [['self-start', 'self-end'], 'self-end'],
    [['place-content-start', 'place-content-end'], 'place-content-end'],
    [['place-items-start', 'place-items-center'], 'place-items-center'],
    [['place-self-start', 'place-self-end'], 'place-self-end'],
    [['tracking-wide', 'tracking-widest'], 'tracking-widest'],
    [['leading-normal', 'leading-relaxed'], 'leading-relaxed'],
    [['list-none', 'list-disc'], 'list-disc'],
    [['list-inside', 'list-outside'], 'list-outside'],
    [['text-opacity-50', 'text-opacity-100'], 'text-opacity-100'],
  ];

  testCases.forEach(([inputs, expected], idx) => {
    it(`handles parameterized merge case #${idx + 1}`, () => {
      expect(cn(inputs)).toBe(expected);
    });
  });
});
