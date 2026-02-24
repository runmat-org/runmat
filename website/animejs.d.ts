declare module "animejs" {
  interface AnimeParams {
    targets?:
      | Element
      | Element[]
      | NodeList
      | string
      | object
      | null;
    [key: string]: unknown;
  }
  function anime(params: AnimeParams): { pause: () => void };
  export default anime;
}
